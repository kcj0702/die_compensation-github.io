"""파일 정리 결과를 MariaDB(카탈로그 + 작업 이력)에 기록하는 저장소.

파일 본체는 로컬/NAS 폴더에만 있고, 여기서는 "무엇이 어디로 갔는지"만 기록한다.
MariaDB 가 설정되지 않았거나 연결에 실패해도 ``core.write_history`` 가 남기는
로컬 감사 로그는 항상 별도로 남으므로, 이 모듈의 실패가 파일 정리 자체를
막지는 않는다 (호출부인 server.py 에서 예외를 잡아 안내만 한다).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from core import Classification, OperationResult

_DB_URL_FILENAME = ".file_db_url"


class DatabaseError(RuntimeError):
    """MariaDB 연결/쿼리 실패 시 발생한다."""


@dataclass(frozen=True)
class RecordBatchResult:
    operation_count: int
    catalog_count: int


def _parse_mysql_url(database_url: str) -> dict[str, Any]:
    parsed = urlsplit(database_url)
    if parsed.scheme not in {"mysql", "mysql+mysqlconnector"}:
        raise DatabaseError(
            "MariaDB 연결 URL은 mysql:// 로 시작해야 합니다."
        )
    database = unquote(parsed.path.lstrip("/"))
    if not parsed.hostname or not parsed.username or not database:
        raise DatabaseError("MariaDB 연결 URL에 host, user, database 이름이 필요합니다.")
    query = parse_qs(parsed.query)
    charset = query.get("charset", ["utf8mb4"])[-1]
    timeout_text = query.get("connect_timeout", ["10"])[-1]
    try:
        connection_timeout = max(1, min(60, int(timeout_text)))
    except ValueError as exc:
        raise DatabaseError("connect_timeout은 1~60 사이 정수여야 합니다.") from exc
    return {
        "host": parsed.hostname,
        "port": parsed.port or 3306,
        "user": unquote(parsed.username),
        "password": unquote(parsed.password or ""),
        "database": database,
        "charset": charset,
        "connection_timeout": connection_timeout,
        "autocommit": False,
    }


class MariaDBRepository:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._config = _parse_mysql_url(database_url)

    @staticmethod
    def _source_key(source: Path) -> str:
        return str(Path(source).resolve())

    def _connect(self) -> Any:
        try:
            import mysql.connector
        except ImportError as exc:
            raise DatabaseError(
                "mysql-connector-python 패키지가 설치되어 있지 않습니다."
            ) from exc
        try:
            return mysql.connector.connect(**self._config)
        except Exception as exc:
            raise DatabaseError(f"MariaDB 연결에 실패했습니다: {exc}") from exc

    def _ensure_schema(self, connection: Any) -> None:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS file_catalog (
                id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                destination_key VARCHAR(500) NOT NULL,
                file_name VARCHAR(255) NOT NULL,
                destination_path TEXT NOT NULL,
                storage_root VARCHAR(500) NULL,
                customer VARCHAR(64) NULL,
                item_no VARCHAR(64) NULL,
                family VARCHAR(64) NULL,
                product_name VARCHAR(255) NULL,
                process VARCHAR(32) NULL,
                category_key VARCHAR(8) NULL,
                category_label VARCHAR(64) NULL,
                updated_at DATETIME NOT NULL,
                PRIMARY KEY (id),
                UNIQUE KEY uq_file_catalog_destination (destination_key)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS file_operations (
                id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                batch_id VARCHAR(64) NOT NULL,
                source_path TEXT NOT NULL,
                destination_path TEXT NOT NULL,
                operation VARCHAR(16) NOT NULL,
                status VARCHAR(16) NOT NULL,
                message VARCHAR(500) NULL,
                item_no VARCHAR(64) NULL,
                category_label VARCHAR(64) NULL,
                created_at DATETIME NOT NULL,
                PRIMARY KEY (id),
                KEY idx_file_operations_batch (batch_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )
        cursor.close()

    def test_and_initialize(self) -> str:
        connection = self._connect()
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()[0]
            cursor.close()
            self._ensure_schema(connection)
            connection.commit()
            return str(version)
        except DatabaseError:
            raise
        except Exception as exc:
            raise DatabaseError(f"MariaDB 초기화에 실패했습니다: {exc}") from exc
        finally:
            connection.close()

    def get_summary(self) -> dict[str, int]:
        connection = self._connect()
        try:
            self._ensure_schema(connection)
            cursor = connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM file_catalog")
            catalog_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM file_operations")
            operation_count = cursor.fetchone()[0]
            cursor.close()
            connection.commit()
            return {"catalogCount": int(catalog_count), "operationCount": int(operation_count)}
        except DatabaseError:
            raise
        except Exception as exc:
            raise DatabaseError(f"MariaDB 조회에 실패했습니다: {exc}") from exc
        finally:
            connection.close()

    def record_batch(
        self,
        *,
        batch_id: str,
        results: list[OperationResult],
        classifications: dict[str, Classification],
        storage_root: Path,
    ) -> RecordBatchResult:
        connection = self._connect()
        try:
            self._ensure_schema(connection)
            cursor = connection.cursor()
            now = datetime.now()
            operation_count = 0
            catalog_count = 0
            for result in results:
                classification = classifications.get(self._source_key(Path(result.source)))
                item_no = classification.item_no if classification else None
                category_label = classification.category_label if classification else None
                cursor.execute(
                    """
                    INSERT INTO file_operations
                        (batch_id, source_path, destination_path, operation, status,
                         message, item_no, category_label, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        batch_id, result.source, result.destination, result.operation,
                        result.status, result.message, item_no, category_label, now,
                    ),
                )
                operation_count += 1

                if result.status == "success":
                    destination_path = Path(result.destination)
                    cursor.execute(
                        """
                        INSERT INTO file_catalog
                            (destination_key, file_name, destination_path, storage_root,
                             customer, item_no, family, product_name, process,
                             category_key, category_label, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            file_name = VALUES(file_name),
                            storage_root = VALUES(storage_root),
                            customer = VALUES(customer),
                            item_no = VALUES(item_no),
                            family = VALUES(family),
                            product_name = VALUES(product_name),
                            process = VALUES(process),
                            category_key = VALUES(category_key),
                            category_label = VALUES(category_label),
                            updated_at = VALUES(updated_at)
                        """,
                        (
                            str(destination_path.resolve()), destination_path.name,
                            str(destination_path), str(storage_root),
                            classification.customer if classification else None,
                            classification.item_no if classification else None,
                            classification.family if classification else None,
                            classification.product_name if classification else None,
                            classification.process if classification else None,
                            classification.category_key if classification else None,
                            classification.category_label if classification else None,
                            now,
                        ),
                    )
                    catalog_count += 1
            cursor.close()
            connection.commit()
            return RecordBatchResult(operation_count=operation_count, catalog_count=catalog_count)
        except DatabaseError:
            connection.rollback()
            raise
        except Exception as exc:
            connection.rollback()
            raise DatabaseError(f"MariaDB 기록에 실패했습니다: {exc}") from exc
        finally:
            connection.close()


def _db_url_path(base_dir: Path) -> Path:
    return base_dir / _DB_URL_FILENAME


def load_database_url(base_dir: Path) -> str:
    path = _db_url_path(base_dir)
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def save_database_url(base_dir: Path, database_url: str) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    _db_url_path(base_dir).write_text(database_url.strip(), encoding="utf-8")


def safe_database_label(database_url: str) -> str:
    if not database_url:
        return "MariaDB 미설정"
    try:
        parsed = urlsplit(database_url)
        database = unquote(parsed.path.lstrip("/")) or "?"
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.hostname}{port}/{database}"
    except ValueError:
        return "MariaDB 연결 정보 확인 필요"
