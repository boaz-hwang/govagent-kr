# GovAgent-KR Temporal Citation Eval

기준일에 맞는 한국 법령 연혁·조문·원문을 Agent가 고르는지 검증하는 작은 실행형
평가 환경이다. 범위는 지방계약법 시행령 한 개, 5개 case family, 20개 item이다.

이 저장소가 판정하는 것은 좁다. 정부 인증, 공공행정 투입 적합성, 완전한 법률 판단,
RL 학습 준비 완료를 의미하지 않는다. 단일 `Overall/PASS`도 계산하지 않는다.

## 현재 상태

- 공식 국가법령정보센터 XML 7개 연혁과 SHA-256을 동결했다.
- 3개 개정 경계에서 5개 family·20개 item을 만들었다.
- exact `(instrument, history, locator, passage hash)` 결합키 채점기와 공격 테스트가 있다.
- Inspect AI Agent Bridge와 외부 네트워크가 없는 Docker SUT를 연결했다.
- fixture oracle, 현재법 치환 ablation, freeze, blind transcript review, 결과 보고 흐름이 있다.
- 20개 후보 gold는 아직 법률 전문가 2인의 독립 검수 전이다.
- 실제 frontier model 60회 실행과 transcript 사람 검토는 아직 하지 않았다.

따라서 현재 결과는 `engineering_environment_ready`까지만 주장할 수 있다. 자세한 범위와
판정 기준은 [pland.md](pland.md), 출처·권리·한계는 [DATASET_CARD.md](DATASET_CARD.md)를 본다.

## 개발 환경 검증

Python 3.12, Docker와 Compose가 필요하다.
v0.1은 일반 설치형 wheel이 아니라 이 checkout에서 실행하는 Inspect task pack이다. 저장소를
clone한 뒤 아래처럼 `uv sync`로 editable 환경을 만들며 PyPI 배포 대상으로 보지 않는다.

```bash
cd gov-agent
uv sync --extra dev --extra inspect --python 3.12
uv run govagent-validate
uv run govagent-diagnostics --output reports/diagnostics.json
docker build --no-cache -t govagent-kr-sut:0.1.0 -f runtime/Dockerfile runtime
GOVAGENT_RUN_DOCKER_SMOKE=1 uv run pytest -q tests/test_inspect_integration.py
uv run govagent-quality --output reports/quality-receipt.json
uv run govagent-report \
  --quality-receipt reports/quality-receipt.json \
  --json-output reports/engineering-readiness.json \
  --markdown-output reports/engineering-readiness.md
```

`govagent-build-fixtures`는 공식 원문을 새로 동결하는 관리 작업에서만 실행한다.
`govagent-build-dataset`은 `data/gold.jsonl`을 다시 `pending` 후보로 덮어쓰므로 SME 검수 후에는
새 평가 cycle을 의도적으로 시작할 때 외에는 실행하지 않는다. 의존성 해시 잠금은
`uv pip compile runtime/requirements.in --python-version 3.12 --generate-hashes --output-file runtime/requirements.lock`
명령으로 생성한다. 평가 중 SUT는 동결
fixture만 read-only로 읽고 법령 웹에는 접근하지 않는다. hosted model 호출은 Inspect host가
provider endpoint로 중계하므로 전체 시스템이 오프라인이라는 뜻은 아니다. `data/gold.jsonl`과
저장소 root는 컨테이너에서 보이지 않는다. 개발 smoke는
pending gold를 명시적으로 허용하지만 그 점수는 분석 결과가 아니다.

## 후보 gold의 전문가 검수

검수 bundle에는 각 case, 후보 gold와 해당 family의 후보 기반 탐색용 원문이 들어 있다.
이 원문 목록은 완전한 법적 근거 목록이 아니다. 두 전문가는 동결 raw XML과 공식 원문을
각자 독립 검색해 후보가 빠뜨린 조문·부칙도 찾아야 한다. 그 뒤 각자 별도 사본의
`review-items.csv`를 작성하고, 차이를 확인해
`adjudication-template.jsonl`에 최종 gold·adjudicator ID·근거를 기록한다.

```bash
cp -f data/gold.jsonl /secure/path/candidate-gold.jsonl

uv run govagent-gold-review export \
  --candidate-gold /secure/path/candidate-gold.jsonl \
  --output /secure/path/gold-review-bundle

uv run govagent-gold-review finalize \
  --candidate-gold /secure/path/candidate-gold.jsonl \
  --review-1 /secure/path/reviewer-1.csv \
  --review-2 /secure/path/reviewer-2.csv \
  --adjudication /secure/path/adjudication.jsonl \
  --review-manifest /secure/path/gold-review-bundle/manifest.json \
  --output-gold data/gold.jsonl \
  --receipt-output /secure/path/gold-review-receipt.json

uv run govagent-validate
```

도구는 20개 전수검토, 서로 다른 reviewer ID, 원본·표시 evidence hash, 명시적 사람
adjudication과 최종 dataset 무결성을 검증한다. ID 문자열은 실제 신원이나 독립성을 인증하지
않으므로, 공개 artifact에는 기관이 부여한 비식별 reviewer ID를 쓰고 실제 신원·독립성은 별도의
접근기록·서명 또는 검수 승인 기록으로 보관해야 한다.

## 분석 가능한 본실행 전 조건

다음 조건이 모두 충족돼야 analysis freeze를 운영상 유효한 것으로 취급한다.

1. 법률 전문가 2인이 20개 item을 독립 검토하고 모든 차이를 adjudication한다.
2. 5-item smoke를 읽고 발견한 task·gold·tool·grader 결함을 수정한다.
3. `gov-agent/` 변경을 commit해 worktree를 깨끗하게 만든다.
4. 실제 SUT 이미지 ID와 requested model·sampling 설정을 확정한다.
5. 그 commit·artifact·image ID로 frozen-compose Docker smoke를 포함한 품질 검사를 통과한다.
6. provider credential은 환경변수로만 주입하고 manifest나 로그에 넣지 않는다.

현재 freeze CLI는 1·3·4·5·6번을 자동 검증한다. 2번의 사람 smoke 판정은 v0.1에서 아직
freeze receipt에 결속하지 않는 운영 게이트다. 다음처럼 family마다 하나씩 고정한 5개 item을
최종 SUT 이미지와 본실행에 쓸 설정으로 1회 실행하고, 자동 점수를 보기 전에 transcript를
검토한다. 이 로그·blind bundle·검토 CSV와 결함 조치 기록을 analysis freeze 옆의 접근 통제된
증거 저장소에 보관한다.

```bash
export GOVAGENT_SUT_IMAGE='sha256:<immutable-image-id>'

uv run inspect eval src/govagent_kr/task.py@govagent_kr \
  -T allow_pending_gold=true \
  -T compose_path=compose.frozen.yaml \
  -T trials=1 \
  --model-spec '{"model":"<provider/model-id>","temperature":0,"reasoning_effort":"high"}' \
  --sample-id 'social-enterprise-direct-award--post_effective,social-enterprise-single-quote--post_effective,statutory-digital-service--post_effective,legacy-innovation-transition--post_effective,material-price-adjustment-transition--post_effective' \
  --epochs 1 \
  --no-epochs-reducer \
  --max-samples 1 \
  --max-connections 1 \
  --max-retries 0 \
  --timeout 120 \
  --attempt-timeout 120 \
  --no-fail-on-error \
  --log-model-api \
  --log-buffer 1 \
  --log-dir /secure/path/smoke-run

uv run govagent-review export \
  --log-dir /secure/path/smoke-run \
  --output /secure/path/smoke-review

# 자동 점수를 보기 전에 primary-review.csv를 작성한 뒤
uv run govagent-review validate \
  --reviews /secure/path/smoke-review/primary-review.csv \
  --identity-map /secure/path/smoke-review/private/identity-map.jsonl
```

`TASK_AMBIGUITY`, `GOLD_ERROR`, `TOOL_OR_ENV_ERROR` 또는 격리 결함이 발견되면 고치고 smoke를
처음부터 다시 한다. 현재 자동 결속의 빈틈 때문에 사람 smoke를 생략해도 CLI 자체는 analysis
freeze를 만들 수 있으므로, 기관 절차에서는 해당 증거의 존재를 별도 승인 조건으로 둔다.

분석 freeze 예시는 다음과 같다. `model-config.json`에는 비밀값이 아니라 실제 실행할
Inspect `GenerateConfig`만 기록한다. 권장 형태는
`{"generate": {...}, "model_args": {}}`다. v0.1은 provider별 우회 capability를 막기 위해
`model_args`를 비워 두며 빈 generation 기본값을 암묵적으로 가정하지 않는다.
프로토콜 고정값 `max_connections=1`, `max_retries=0`, `timeout=120`,
`attempt_timeout=120`도 `generate`에 명시한다. `extra_headers`는 credential 유출을 막기 위해
금지되며 인증정보는 provider 표준 환경변수로만 전달한다. 시작 형식은
[`model-config.example.json`](model-config.example.json)에 있다.

```bash
docker image inspect govagent-kr-sut:0.1.0 --format '{{.Id}}'
docker image save -o /secure/path/govagent-kr-sut-0.1.0.tar govagent-kr-sut:0.1.0
shasum -a 256 /secure/path/govagent-kr-sut-0.1.0.tar

uv run govagent-quality \
  --container-image 'sha256:<immutable-image-id>' \
  --output /absolute/path/analysis-quality-receipt.json

uv run govagent-freeze \
  --mode analysis \
  --requested-model '<provider/model-id>' \
  --model-config /absolute/path/model-config.json \
  --container-image 'sha256:<immutable-image-id>' \
  --quality-receipt /absolute/path/analysis-quality-receipt.json \
  --gold-review-receipt /secure/path/gold-review-receipt.json \
  --output /absolute/path/analysis-freeze.json

uv run govagent-verify-freeze \
  /absolute/path/analysis-freeze.json \
  --require-analysis
```

freeze receipt와 model config는 실행 증거 저장소에 보관한다. provider key나 review blinding
key는 Git에 넣지 않는다. 로컬 image ID만으로 다른 장비에서 복원할 수 없으므로 OCI registry
digest 또는 위 image archive와 그 SHA-256도 함께 보관한다. freeze 후 critical artifact가 바뀌면 기존 run은 분석에서 제외하고
20개 전체를 3회 다시 실행한다.

## Frontier 60회 실행

Task가 freeze의 model ID·generation config·model args를 직접 읽는다. CLI에서 model이나
sampling 옵션을 다시 덮어쓰지 않는다. 한 번의 동결된 로그만 최종 보고 입력으로 사용하며
sample retry는 허용하지 않는다.

```bash
export GOVAGENT_SUT_IMAGE='sha256:<immutable-image-id>'

uv run inspect eval src/govagent_kr/task.py@govagent_kr \
  -T compose_path=compose.frozen.yaml \
  -T freeze_manifest=/absolute/path/analysis-freeze.json \
  --epochs 3 \
  --no-epochs-reducer \
  --max-samples 1 \
  --max-connections 1 \
  --max-retries 0 \
  --timeout 120 \
  --attempt-timeout 120 \
  --no-fail-on-error \
  --log-model-api \
  --log-buffer 1 \
  --log-dir reports/runs/frontier
```

`compose.frozen.yaml`은 tag가 아니라 freeze에 기록한 `sha256:` image ID만 받도록 운용한다.
기관 환경에서는 SUT 컨테이너 격리와 별도로 host egress를 provider endpoint allowlist로
제한해야 한다.

## Transcript 검토와 최종 결과

grader를 보기 전에 주 검토자가 60개 transcript를 전부 판정한다. 두 번째 검토자는 오류
label이 있는 전부와 결정적으로 뽑힌 clean 10개를 독립 검토한다. 차이는 사람이 adjudication한
뒤에만 grader를 공개한다.

```bash
uv run govagent-review export \
  --log-dir reports/runs/frontier \
  --output /secure/path/primary-bundle

# primary-bundle/primary-review.csv를 주 검토자가 작성한 뒤
uv run govagent-review validate \
  --reviews /secure/path/primary-bundle/primary-review.csv \
  --identity-map /secure/path/primary-bundle/private/identity-map.jsonl

uv run govagent-review export-secondary \
  --primary-reviews /secure/path/primary-bundle/primary-review.csv \
  --bundle /secure/path/primary-bundle \
  --output /secure/path/secondary-bundle

# 다른 검토자가 secondary-review.csv를 작성한 뒤
uv run govagent-review validate-secondary \
  --reviews /secure/path/secondary-bundle/secondary-review.csv \
  --primary-reviews /secure/path/primary-bundle/primary-review.csv \
  --bundle /secure/path/primary-bundle \
  --secondary-bundle /secure/path/secondary-bundle

uv run govagent-review disagreements \
  --primary-reviews /secure/path/primary-bundle/primary-review.csv \
  --secondary-reviews /secure/path/secondary-bundle/secondary-review.csv \
  --bundle /secure/path/primary-bundle \
  --secondary-bundle /secure/path/secondary-bundle \
  --output /secure/path/disagreements.json

uv run govagent-review export-adjudication \
  --disagreements /secure/path/disagreements.json \
  --output /secure/path/adjudications-template.jsonl

cp -f /secure/path/adjudications-template.jsonl /secure/path/adjudications.jsonl

# 제3의 검토자가 grader를 보지 않은 채 adjudications.jsonl을 완성한 뒤
uv run govagent-review validate-adjudication \
  --disagreements /secure/path/disagreements.json \
  --adjudications /secure/path/adjudications.jsonl

uv run govagent-review reveal \
  --reviews /secure/path/primary-bundle/primary-review.csv \
  --bundle /secure/path/primary-bundle \
  --secondary-reviews /secure/path/secondary-bundle/secondary-review.csv \
  --secondary-bundle /secure/path/secondary-bundle \
  --disagreements /secure/path/disagreements.json \
  --adjudications /secure/path/adjudications.jsonl \
  --output /secure/path/revealed-reviews.jsonl
```

최종 보고기는 정확히 `20 items x 3 epochs`, freeze/model 결합, 사람 검토의 완전성, 공개된
자동 점수를 재검증한다. 결론은 `go_expand_eval`, `redesign`, `stop` 중 사람이 선택한다.
`go_expand_eval`은 60회가 모두 분석 가능·환경 유효하고 미결 사람 판정이 없을 때만 허용한다.
그 결정도 이 평가를 확장할 근거일 뿐 RL 또는 배포 승인이 아니다.

```bash
uv run govagent-final-report \
  --freeze /absolute/path/analysis-freeze.json \
  --log-dir reports/runs/frontier \
  --review-bundle /secure/path/primary-bundle \
  --primary-reviews /secure/path/primary-bundle/primary-review.csv \
  --secondary-bundle /secure/path/secondary-bundle \
  --secondary-reviews /secure/path/secondary-bundle/secondary-review.csv \
  --revealed-reviews /secure/path/revealed-reviews.jsonl \
  --disagreements /secure/path/disagreements.json \
  --adjudications /secure/path/adjudications.jsonl \
  --decision redesign \
  --decision-note '사람이 작성한 근거' \
  --json-output /secure/path/mvp-result.json \
  --markdown-output /secure/path/mvp-result.md
```

disagreement가 없으면 export한 adjudication template은 빈 파일이다. 위 예시처럼 그 빈 파일도
`reveal`과 `govagent-final-report`에 그대로 넘기면 한 가지 실행 경로를 유지할 수 있다.
disagreement가 하나라도 있으면 독립적인 제3 검토자의 grader-blind adjudication을 검증하기
전에는 grade를 공개할 수 없다. review bundle의 private key와 identity map은 transcript
검토자에게 공개하지 않고 접근 통제된 위치에 둔다.
