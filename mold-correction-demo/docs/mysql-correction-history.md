# HeidiSQL로 중앙 보정 이력 DB 설정

HeidiSQL은 데이터베이스 서버가 아니라 MySQL 또는 MariaDB 서버에 접속해
데이터베이스를 관리하는 Windows 프로그램입니다. 먼저 사내의 한 PC나 서버에
MySQL/MariaDB 서버가 실행 중이어야 합니다.

각 작업자 PC의 ADC 백엔드는 `AJIN_CORRECTION_DB_URL`에 지정된 동일한 서버를
사용합니다. 이 값이 없으면 개발과 테스트를 위해 기존 로컬 SQLite 파일을
사용합니다.

## 1. HeidiSQL에서 서버 접속

1. HeidiSQL을 실행하고 왼쪽 아래의 `신규`를 누릅니다.
2. 네트워크 유형을 `MariaDB or MySQL (TCP/IP)`로 선택합니다.
3. 서버의 IP 주소, 사용자, 비밀번호와 포트 `3306`을 입력합니다.
4. `열기`를 눌러 접속합니다.

HeidiSQL만 설치되어 있고 접속할 서버가 없다면 MySQL Server 또는 MariaDB
Server를 한 대의 중앙 PC에 먼저 설치해야 합니다.

## 2. HeidiSQL 쿼리 탭에서 DB와 계정 생성

관리자 계정으로 접속한 뒤 `쿼리` 탭을 열고 아래 내용을 붙여 넣습니다.
비밀번호와 허용할 사내 네트워크 대역은 실제 환경에 맞게 변경한 후 `F9`로
실행합니다.

```sql
CREATE DATABASE ajin_adc
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

CREATE USER 'adc_app'@'192.168.%'
  IDENTIFIED BY 'CHANGE_THIS_PASSWORD';

GRANT SELECT, INSERT, DELETE, CREATE, ALTER, INDEX
  ON ajin_adc.*
  TO 'adc_app'@'192.168.%';

FLUSH PRIVILEGES;
```

ADC가 처음 연결되면 `correction_history` 테이블과 인덱스를 자동 생성합니다.
테이블은 HeidiSQL 왼쪽 트리의 `ajin_adc` 데이터베이스에서 확인할 수 있습니다.

## 3. 각 작업자 PC에 연결 정보 설정

PowerShell에서 다음 명령을 한 번 실행합니다.

```powershell
setx AJIN_CORRECTION_DB_URL "mysql://adc_app:CHANGE_THIS_PASSWORD@192.168.0.20:3306/ajin_adc?charset=utf8mb4"
```

기존 터미널과 ADC 프로그램을 완전히 종료한 후 다시 실행합니다. 모든 PC에
같은 URL을 설정하면 동일한 보정 이력을 조회합니다.

비밀번호에 `@`, `:`, `/`, `#` 같은 문자가 있으면 URL 인코딩해야 합니다.
TLS 인증서를 사용하는 서버는 URL에 `ssl_ca` 경로를 추가할 수 있습니다.

```text
mysql://adc_app:password@db.company.local:3306/ajin_adc?charset=utf8mb4&ssl_ca=C%3A%5Ccerts%5Ccompany-ca.pem
```

## 4. Python 드라이버 설치

```powershell
.venv\Scripts\python.exe -m pip install -r mold-correction-demo\ui\backend\requirements.txt
```

접속 비밀번호를 소스 코드나 Git 저장소에 커밋하지 않습니다.
