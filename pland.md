# GovAgent-KR Temporal Citation Eval MVP

> 기준일에 맞는 한국 법령 연혁과 조문을 AI Agent가 선택·인용하는지 검증하는 최소 평가 환경

- 기준일: 2026-08-25
- 개발기간: 10영업일
- 규모: 5 case families, 20 items, 60 frontier-model runs
- 상태: engineering environment 구현 완료, 법률 전문가 검수·frontier run 전
- 결과물의 지위: 개발용 evaluation environment
- 아닌 것: RL 학습환경, 정부 인증, 법률의견, 공공행정 전체 benchmark

---

## 0. 결정

지금 만들 것은 `한국 공공행정 Agent 표준`도 `RL 플랫폼`도 아니다.

다음 질문 하나에 답하는 작은 평가 환경을 만든다.

> **Agent가 현재법을 무심코 가져오지 않고, 문제에 명시된 기준일에 유효한 지방계약법 시행령의 연혁과 조문을 찾아 정확히 인용할 수 있는가?**

MVP의 목적은 제품을 출시하는 것이 아니라 세 가지를 검증하는 것이다.

1. 실제 프론티어 Agent가 이 좁은 과제에서 실패하는가
2. 그 실패를 deterministic grader가 정확히 잡는가
3. 실패가 모델 문제인지 task·gold·grader·tool 문제인지 transcript를 읽고 구분할 수 있는가

셋 중 하나라도 확인되지 않으면 범위를 늘리지 않는다.

### 한 줄 원칙

> **작아서 모든 사례, gold, grader 판정과 60개 transcript를 사람이 직접 검토할 수 있어야 한다.**

### 현재 구현 결과

2026-08-26 기준으로 사람이 아닌 공학 작업은 구현돼 있다.

| 영역 | 구현 상태 | 분석 사용 조건 |
| --- | --- | --- |
| 공식 corpus | 7개 XML·정규화 원문·manifest·SHA-256 동결 | 전문가가 경계·본문·부칙 확인 |
| task data | 3개 경계, 5 families, 20 items | SME 2인 전수검토와 adjudication |
| grader | strict JSON, 결론, 4-tuple citation, 기권·오류 분리 | gold 확정 뒤 freeze |
| 실행 환경 | Inspect Agent Bridge, Docker network none, gold 미마운트 | 고정 provider/model로 smoke |
| 진단 | fixture oracle 20/20, current-version ablation 9/20 실패 | 모델 성능으로 해석하지 않음 |
| 증거 흐름 | quality·gold·5-item smoke receipt, analysis freeze, blind 1·2차 검토, 최종 report | 실제 smoke·60 runs와 사람 검토 필요 |

분석 freeze는 현재 commit·critical artifact set·정확한 `sha256` SUT image와 그 이미지로
수행한 Docker smoke를 하나의 quality receipt로 묶는다. pending gold, mutable image tag, dirty
worktree 또는 낡은 quality receipt 중 하나라도 있으면 본실행을 거부한다.

아직 만들지 않은 증거도 명확하다. 현재 gold 20건은 `pending`이고 provider credential이 없어
frontier 5-item smoke와 60 runs는 실행하지 않았다. transcript 사람 검토와
`Go / Redesign / Stop` 결정도 없다.
따라서 구현 완료는 `평가 환경의 공학적 준비`를 뜻하며 평가 결과 완료를 뜻하지 않는다.

---

## 1. 왜 평가 환경부터 만드는가

RL 환경은 task와 grader가 이미 믿을 만하다는 전제 위에 선다. 그 전제가 틀리면 모델은 실제 업무능력이 아니라 grader의 허점을 학습한다.

따라서 순서는 다음과 같다.

```text
작은 eval environment
        ↓
frontier model 반복실행
        ↓
모델 실패 / 평가 실패 분리
        ↓
task와 grader 타당성 확인
        ↓
필요할 때만 task 확대
        ↓
충분한 학습신호가 확인된 뒤 RL environment 검토
```

v0.1에서 하지 않는 일:

- scalar RL reward 설계
- trainer·GPU 학습 연동
- production workflow log 수집
- 자동 task 생성
- trajectory를 이용한 fine-tuning
- recursive self-improvement loop

다만 향후 변환할 수 있도록 task ID, 환경 버전, tool action·observation, grader version, 종료사유와 transcript는 구조화해 남긴다.

---

## 2. 선행 구현과 이 MVP의 위치

범용 Agent 평가와 RL 환경 기반은 이미 존재한다.

| 선행 구현 | 이미 하는 일 | 이 MVP가 추가하는 것 |
| --- | --- | --- |
| NIA AI 에이전트 안전·신뢰성 검증 사업[1] | 범용 Agent benchmark, trajectory, MCP 검증체계 구축 | 한국 법령의 좁은 시점 선택 task와 gold |
| UK AISI Inspect AI[2] | task, tool, scorer, sandbox, transcript | 지방계약법 시행령용 task pack |
| NIST Agentic Evaluation Probes[3] | claim-source audit trail과 citation 평가 | 한국 공식 법령 연혁 ID와 기준일 정답 |
| KBL·KoBLEX[4][5] | 한국 법률 QA·RAG와 근거 조문 평가 | 과거 연혁 선택과 통제된 tool environment |
| FiscalQA Pro[6] | 시간축 법률 benchmark와 provenance | 한국 법령 형식과 전문가 검수 사례 |
| OpenEnv·Verifiers[7][8] | Agentic RL environment와 reward/training 연동 | MVP 통과 후 고려할 export 대상 |

따라서 새 평가 엔진을 만들지 않는다. Inspect AI를 실행기와 transcript 원본으로 사용한다.[2]

이 MVP의 신규성도 과장하지 않는다.

> 시간축 법률 benchmark 자체가 새로운 것은 아니다. 한국 공식 법령 연혁과 작은 전문가 검수 task를 실행 가능한 환경으로 만드는 것이 이번 검증 대상이다.

---

## 3. 검증할 가설

### H1. 시간축 실패가 실제로 존재한다

gold answer의 연혁만 현재 최신 연혁으로 바꾸는 `current-version ablation`이 시행 전·후 temporal trap에서 실패한다. 이는 모델 성능이 아니라 case가 현재법 오용을 실제로 구분하는지 확인하는 진단이다.

### H2. 프론티어 Agent도 같은 실패를 보인다

동일한 corpus와 tool을 제공해도 강한 Agent가 반복적으로 잘못된 연혁·조문을 고르는지 탐색한다. `2개 case family`는 확대 가치를 판단하는 내부 진단값이지 일반적인 품질기준이 아니다.

### H3. 실패를 결정적으로 판정할 수 있다

정답 `history_id`, 조문 locator와 exact quote만으로 core success를 LLM judge 없이 판정할 수 있다.

### H4. 사람이 환경 결함을 찾아 고칠 수 있다

60개 transcript를 전수 검토해 모델 실패와 task·gold·grader·tool·infra 실패를 구분할 수 있다.

### 반증될 경우

- 프론티어 Agent가 모두 맞히면 task가 너무 쉽다.
- 전문가가 gold에 합의하지 못하면 task가 너무 모호하다.
- LLM judge가 없으면 채점할 수 없다면 scope가 너무 넓다.
- transcript만으로 실패 원인을 구분할 수 없다면 환경 관측성이 부족하다.

어느 경우든 사례 수부터 늘리지 않는다.

---

## 4. MVP 범위

### 포함

- 법령: `지방자치단체를 당사자로 하는 계약에 관한 법률 시행령` 한 개
- 자료: 공식 연혁과 관련 부칙
- 법령 변경 경계: 실제 개정 3개
- case family: 5개
- item: family당 4개, 총 20개
- frontier SUT: 1개
- frontier trial: item당 3회, 총 60 runs
- environment diagnostics: fixture oracle 1개, current-version ablation 1개
- 실행기: Inspect AI
- SUT container 외부 네트워크: 차단; Inspect host의 model-provider egress는 별도 allowlist

### 제외

- 지방계약법·시행규칙과의 위계 판단
- 조례·행정규칙·예규·판례·유권해석
- 실제 계약·입찰 데이터
- 법적 결론이나 설득력 있는 논증
- 개인정보, 기관 내부자료와 실제 담당자
- HWPX, 결재, 승인과 write action
- 여러 모델 leaderboard
- 비공개 인증셋
- policy pack과 범용 trace schema

이 범위로는 실제 지방계약 업무의 적법성을 평가할 수 없다. 오직 `명시된 날짜에 맞는 시행령 연혁·조문·문구 선택`만 평가한다.

---

## 5. Case 구성

### 5.1 변경 경계 3개

공식 국가법령정보센터 XML과 개정문을 대조해 다음 세 경계를 동결했다.

| 경계 | 직전 / 직후 `lsiSeq` | 평가하는 변화 |
| --- | --- | --- |
| 2018-07-24 | `201260` / `204108` | 사회적기업 수의계약과 1인 견적 예외 신설 |
| 2020-11-03 | `221951` / `222683` | 디지털서비스 신설, 종전 혁신제품 경과조치 |
| 2025-07-08 | `269761` / `272723` | 특정규격 자재 가격증감 기준 15%→10%와 미완료 계약 적용례 |

각 원본 XML·정규화 원문·출처·유효기간·SHA-256은 fixture manifest에 보관한다. 이 선정도
현재는 engineering candidate이며, 법률 전문가 2인의 독립 검토와 adjudication 전에는
확정 gold로 부르지 않는다.

### 5.2 Case family 5개

- 변경 경계 A에 의존하는 family 2개
- 변경 경계 B에 의존하는 family 2개
- 변경 경계 C에 의존하는 family 1개

각 family는 네 item으로 만든다.

1. 시행 직전 날짜
2. 시행 직후 날짜
3. 현재법을 고르게 유도하는 decoy 표현
4. 기준일은 유지하되 적용요건에 필요한 사실이 빠져 `insufficient_facts`가 정답인 변형

총 20개 item이다. 이 수는 통계적으로 일반화하기 위한 표본이 아니라 환경의 타당성을 확인하기 위한 engineering sample이다.

### 5.3 Gold 작성

법률 전문가 2인이 각 item을 독립 검토한다.

Gold 필드:

```text
expected_status
expected_conclusion
required_citations[] {
  instrument_id,
  history_id,
  provision_locator,
  exact_passage_hash
}
allowed_citations[] {
  instrument_id,
  history_id,
  provision_locator,
  exact_passage_hash
}
forbidden_history_ids[]
required_missing_fact_keys[]
controlling_event
adjudication_note
```

두 전문가가 다르면 논의해 확정한다. 해결되지 않은 item은 제외한다. 20개를 맞추기 위해 모호한 사례를 남기지 않는다.

---

## 6. Task와 환경 계약

### 6.1 입력

```json
{
  "task": "kr-law-temporal-citation",
  "question": "기준일 당시 적용되는 조문을 찾으시오.",
  "facts": ["..."],
  "instrument_id": "<official-stable-id>",
  "controlling_event": {
    "kind": "bid_notice",
    "date": "YYYY-MM-DD",
    "timezone": "Asia/Seoul"
  }
}
```

법 적용을 지배하는 사건 종류와 날짜는 명시한다. 어떤 사건일이 법적으로 우선하는지를 해석하는 문제는 MVP 범위 밖이다.

`controlling_event.date`는 모든 유효 item에서 required다. `insufficient_facts` item도 날짜는 제공하고 계약금액 등 적용요건에 필요한 domain fact만 누락한다.

### 6.2 Tool

Agent에게 네 개만 제공한다.

```text
law.list_versions(instrument_id)
law.search_provisions(history_id, query)
law.get_provision(history_id, provision_locator)
law.get_supplementary_provisions(history_id)
```

도구는 공식 연혁·시행일·원문을 반환하지만 사건의 정답 연혁을 직접 알려주지 않는다.

### 6.3 출력

```json
{
  "status": "answered",
  "conclusion": "affirmative",
  "selected_provisions": [
    {
      "instrument_id": "...",
      "history_id": "...",
      "provision_locator": "...",
      "quote": "..."
    }
  ],
  "missing_facts": [],
  "explanation": "..."
}
```

`status`는 세 값만 허용한다.

```text
answered
insufficient_facts
no_applicable_provision
```

`conclusion`은 질문에 적힌 명제의 실체적 방향을 나타내는 세 값만 허용한다.

```text
affirmative
negative
indeterminate
```

`answered`는 `affirmative` 또는 `negative`, `insufficient_facts`는 `indeterminate`,
`no_applicable_provision`은 `negative`여야 한다. 이 필드를 별도로 채점하지 않으면 정확한
조문을 붙이고 정반대 결론을 적어도 통과하므로 필수 계약으로 둔다.

`explanation`은 transcript에는 남기지만 core score에는 사용하지 않는다.

provider timeout과 tool·sandbox crash는 Agent가 제출하는 status가 아니다. Inspect harness의
`episode_outcome`으로 별도 기록하고 주 capability 분석에서 제외한다. 반면 model/tool budget
소진과 최종 제출 누락은 SUT가 평가를 회피하지 못하도록 고정 60회 `attempt_success` 분모에서
실패로 센다.

### 6.4 실행 제한

```yaml
timezone: Asia/Seoul
wall_clock_seconds: 180
model_requests: 12
tool_calls: 24
submission_attempts: 1
trials_per_item: 3
network: none
provider_web_search: false
provider_code_execution: false
remote_mcp: false
score_feedback_during_run: false
```

gold와 grader metadata는 Agent sandbox 밖에 둔다. clock, corpus, tool response와 budget은 모든 run에서 동일하다.

SUT는 Inspect native in-process agent가 아니라 `sandbox_agent_bridge` container에서 실행한다.
container는 `network_mode:none`, read-only root filesystem, capability drop 상태이며 gold와 저장소
root를 mount하지 않는다. hosted model 요청만 Inspect host의 loopback proxy가 중계한다.
container의 외부 route·gold 경로·공개 fixture mount를 매 episode preflight로 검사한다.

Inspect host 자체의 OS-level provider endpoint allowlist는 repository code만으로 강제하지 못한다.
기관 환경에서 실행할 때는 별도 egress gateway로 provider endpoint만 허용하고 그 정책 revision을
freeze receipt 밖의 운영 증거로 보관한다. MVP의 자동 격리 보증 범위는 SUT container까지다.

---

## 7. Grader

### 7.1 Core 판정

한 episode의 `core_success`는 다음을 모두 만족할 때만 1이다.

1. output schema가 유효하다.
2. `status`가 gold와 일치한다.
3. `conclusion`이 gold와 일치한다.
4. 모든 `required_citations` tuple을 선택했다.
5. 선택한 모든 citation tuple이 `required_citations` 또는 `allowed_citations`에 속한다.
6. forbidden history를 선택하지 않았다.
7. 각 quote의 canonical hash가 같은 tuple에 결합된 `exact_passage_hash`와 일치한다.

그 외에는 0이다.

`core_success`는 MVP 분석용 binary metric이다. RL reward로 사용한다고 아직 결정하지 않는다.

grader는 `instrument_id + history_id + provision_locator + canonical passage hash`를 하나의 결합키로 비교한다. 네 필드를 따로 비교하지 않는다. quote 비교는 versioned canonicalization 뒤 수행한다. v0.1은 Unicode NFC, 줄바꿈 LF 통일, 행 끝 공백 제거까지만 허용하고 구두점·단어·숫자는 바꾸지 않는다. 짧은 공통문구나 빈 문자열로 통과하지 못하도록 substring 포함 여부가 아니라 gold tuple에 등록된 전체 passage hash와 비교한다.

공식 XML이 `마목` 하나의 평문 안에 `1)~6)`을 넣는 경우 parser가 원 aggregate passage는
보존하면서 `#본문`, `#3)`, `#8)` 같은 synthetic sublocator를 추가한다. 이는 공식 조문번호를
새로 만든 것이 아니라 exact contiguous passage를 과잉인용 없이 가리키는 v0.1 parser
extension이며 manifest의 parser version에 묶인다.

### 7.2 보조 지표

- temporal history accuracy
- required provision recall
- selected provision precision
- quote integrity
- forbidden-current-version rate
- correct abstention rate
- tool calls, latency, tokens

단일 `Overall 95.1 PASS`는 만들지 않는다. 20개 item의 점수로 `공공행정 투입 가능`을 판정하지 않는다.

### 7.3 Grader test

각 grader에는 최소 다음 fixture를 둔다.

- 완전 정답
- 올바른 조문이지만 잘못된 연혁
- 필수 정답과 함께 같은 locator의 비금지·비허용 연혁을 추가한 과잉 인용
- 올바른 연혁이지만 잘못된 항·호
- 다른 연혁에서 가져온 정확한 문구
- 인용문 일부 변조
- 전부 기권
- schema 오류
- 존재하지 않는 source·history ID

golden grader test는 모두 통과해야 한다. grader를 한 줄이라도 바꾸면 60개 transcript를 전부 재채점한다.

---

## 8. 실행과 Transcript 검토

### 8.1 진단도구와 SUT 동결

두 deterministic diagnostic은 모델 비교용 baseline이 아니다.

- `fixture oracle`: gold submission을 재생해 tool·fixture·grader가 end-to-end로 20/20 동작하는지 확인
- `current-version ablation`: gold의 locator는 유지하되 연혁과 passage를 최신 버전으로 바꿔 temporal trap이 실제로 실패하는지 확인

frontier SUT는 본실행 전에 다음 manifest로 동결한다.

- agent code commit와 container digest
- system·developer prompt hash
- requested model ID와 provider가 반환한 model ID
- reasoning·sampling configuration
- tool contract와 budget
- Inspect·environment·grader version

5개 item으로 smoke rollout을 먼저 수행한다. transcript를 grader-blind로 읽고
task·gold·tool·grader·격리를 확인한다. smoke receipt는 고정 5개 sample ID, 1회 실행,
smoke freeze, Inspect log, blind bundle, 사람 검토 CSV, model config, immutable image와 현재
artifact set을 함께 hash로 묶는다. 사람 판정과 동결 grader가 5건 모두 일치하고 환경·설계
결함이 없을 때만 receipt가 발급된다. analysis freeze는 이 receipt를 필수 입력으로 받고 현재
quality·gold-review receipt, commit, corpus와 다시 대조한다. 어느 하나라도 달라지면 본실행을
거부한다. reviewer ID의 실제 신원·독립성은 저장소가 인증하지 않으므로 기관 승인기록을 별도로
보관한다.

freeze 후 environment, gold, tool, prompt contract나 grader가 바뀌면 기존 본실행 60 runs는 분석에서 제외하고 전부 다시 실행한다. 일부 run만 새 버전으로 교체해 합산하지 않는다.

### 8.2 본실행량

```text
fixture oracle             20 items × 1 = 20
current-version ablation   20 items × 1 = 20
frontier SUT               20 items × 3 = 60
```

핵심 분석 대상은 freeze 후 frontier SUT의 60 runs다. 모델을 여러 개 비교하지 않는다. 첫 SUT에서 유용한 실패 신호가 없을 때만 다음 MVP에서 두 번째 SUT를 검토한다.

### 8.3 전수검토

60개 transcript를 사람이 전부 읽는다. 실패뿐 아니라 높은 점수를 받은 transcript도 읽어 우연한 정답과 grader gaming을 찾는다.

각 run은 자동 grader 점수를 가린 상태에서 먼저 사람이 판정한다.

```text
human_task_verdict: correct | incorrect | indeterminate
grader_agrees: true | false | not_applicable
environment_valid: true | false
sut_error: true | false | indeterminate
labels: []
primary_cause: <one label or null>
```

`labels`에는 여러 원인을 함께 기록하고, 사후 분석용 `primary_cause`만 하나 선택한다. 허용 label은 다음과 같다.

```text
SUT_ERROR
TASK_AMBIGUITY
GOLD_ERROR
TOOL_OR_ENV_ERROR
INFRA_ERROR
```

`GRADER_ERROR`는 grader를 가린 1·2차 검토와 adjudication에서 직접 붙이지 않는다. 사람 판정이
고정된 뒤 자동 점수를 공개해 false positive·false negative와 `grader_agrees=false`를
결정적으로 계산한다.

판정 원칙:

- 모델이 필요한 정보와 도구를 받았는데 잘못 선택했다면 `SUT_ERROR`
- 질문이 두 가지로 읽히면 `TASK_AMBIGUITY`
- 전문가 정답이 잘못됐으면 `GOLD_ERROR`
- transcript상 답은 맞는데 점수가 틀리면 공개 뒤 grader disagreement로 자동 집계
- tool이 잘못된 자료를 반환했으면 `TOOL_OR_ENV_ERROR`
- provider timeout 등은 `INFRA_ERROR`

환경 문제를 고친 run은 원래 결과를 덮어쓰지 않는다. smoke 중이면 freeze 전에 다시 확인하고, 본실행 뒤라면 60 runs 전체를 새 environment·grader version으로 다시 실행한다.

오류가 없으면 `labels=[]`, `primary_cause=null`로 둔다. 주 검토자가 60개를 모두 읽고, 두 번째 검토자가 label이 하나라도 있는 모든 run과 무작위 clean run 10개를 독립 검토한다. 두 검토자 모두 첫 판정 때 자동 grader 결과를 보지 않는다. human verdict를 고정한 뒤 grader 결과를 공개해 `grader_agrees`를 계산한다. 판정이 다르면 transcript를 함께 보고 adjudication한다.

### 8.4 필수 기록

- task·case·environment·grader version
- corpus snapshot hash
- prompt와 tool schema hash
- requested/returned model ID
- sampling parameter
- tool request와 result
- final submission
- token·time·cost
- blind human verdict, grader agreement, environment validity, labels와 reviewer note

Agent가 스스로 작성한 trace를 원본으로 믿지 않는다. Inspect가 model·tool 경계에서 관측한 transcript를 원본으로 삼는다. OpenTelemetry export는 MVP 이후 선택사항이며 감사원장으로 간주하지 않는다.[9]

---

## 9. 성공·실패 기준

### 환경이 작동했다

- fixture oracle이 20/20을 맞힌다.
- current-version ablation이 temporal item에서 최소 3건 실패한다.
- 전문가 adjudication 후 unresolved gold가 0건이다.
- golden grader test가 100% 통과한다.
- frozen transcript를 재채점하면 결과가 bit-for-bit 동일하다.
- 60개 transcript에 blind human verdict, grader agreement, environment validity, labels와 검토 note가 있다.
- 외부 웹과 gold 접근이 차단된다.

### 다음 eval 단계로 진행한다

위 조건에 더해 60회가 모두 analysis-eligible이고 사람이 environment-valid로 확인했으며,
`indeterminate` 판정이 하나도 남지 않은 상태에서 frontier SUT의 비혼재 진짜 `SUT_ERROR`가
최소 2개 case families에서 발견되면 다음 단계 검토 대상으로 삼는다. task·gold·grader·환경
원인이 함께 붙은 run은 이 신호로 세지 않는다. 이 수치는 작은 MVP의 학습신호 탐색용
heuristic이지 모델 품질 또는 통계적 합격선이 아니다.

이때도 곧바로 RL을 시작하지 않는다. 실패 유형을 반영한 case family를 20~30개로 늘릴 가치가 생겼다고만 판정한다.

### 다시 설계한다

- frontier SUT가 60회 모두 맞힌다: 너무 쉽거나 이미 포화됐다.
- 정답처럼 보이는 출력이 grader를 반복적으로 속인다.
- `TASK_AMBIGUITY`, `GOLD_ERROR`, `GRADER_ERROR`가 SUT_ERROR보다 많다.
- 부칙 적용을 두 전문가가 합의하지 못한다.
- fixture oracle도 실패한다.

### 중단한다

- 한 법령 안에서 명확한 변경 경계 3개를 만들 수 없다.
- core 판정에 LLM judge나 주관적 법률논증 평가가 반드시 필요하다.
- 공식 연혁을 고정·재현할 수 없다.
- 작은 환경에서도 유지보수할 전문가를 확보할 수 없다.

---

## 10. 산출물

```text
gov-agent/
├── LICENSE
├── README.md
├── profile.yaml
├── model-config.example.json
├── compose.yaml
├── compose.frozen.yaml
├── DATASET_CARD.md
├── fixtures/
│   └── local-contract-decree/<history>/
├── data/
│   ├── cases.jsonl
│   └── gold.jsonl        # Agent sandbox에서 격리
├── runtime/
│   ├── Dockerfile
│   ├── agent.py
│   ├── law_store.py
│   ├── requirements.in
│   └── requirements.lock
├── src/govagent_kr/
│   ├── task.py
│   ├── law_tools.py
│   ├── scorer.py
│   ├── diagnostics.py
│   ├── gold_review.py
│   ├── model_config.py
│   ├── quality.py
│   ├── freeze.py
│   ├── review.py
│   └── report.py
├── tests/
│   └── test_*.py
└── reports/
    ├── diagnostics.json
    ├── quality-receipt.json
    └── engineering-readiness.{json,md}
```

필수 산출물은 여덟 개다.

1. 동결 corpus와 hash manifest
2. 20개 case, 후보 gold와 2인 전문가 검수·adjudication receipt
3. Inspect task
4. 네 개의 mock law tools
5. deterministic grader와 golden tests
6. fixture oracle과 current-version ablation
7. 60개 frontier transcript와 전수 root-cause review — 외부 실행·사람 검토 대기
8. `Go / Redesign / Stop` 결과 보고서 — 7번 증거가 있어야 생성 가능

새 CLI, dashboard, badge, website와 여러 adapter는 만들지 않는다.

---

## 11. 10영업일 일정

| 일자 | 작업 | 종료조건 |
| --- | --- | --- |
| 1~2일 | 개정 경계 3개 선정, 공식 연혁 동결 | 전문가 2인이 경계와 적용일에 합의 |
| 3~4일 | 5 families·20 items와 gold 작성 | unresolved item 0건 |
| 5일 | sandbox Agent Bridge, mock tools, grader | 격리 canary와 grader tests 통과 |
| 6일 | 두 diagnostic과 5-item smoke rollout | transcript를 읽고 환경 결함 목록 확정 |
| 7일 | 결함 수정, SUT·environment·gold·grader freeze | version manifest와 freeze receipt |
| 8일 | freeze된 frontier SUT 60 runs | 60개 동일 protocol run 확보 |
| 9일 | blind 전수검토와 2차 검토 | 모든 run에 독립 판정 필드 기록 |
| 10일 | reviewer adjudication, replay, 결과보고 | Go / Redesign / Stop 판정 |

필요 인력:

- 평가 엔지니어 1명
- 법률 전문가 2명, 각 8~12시간

10일 안에 끝나지 않으면 기능을 추가하지 않고 case 수 또는 법령 범위를 더 줄인다.

Day 8 이후 freeze 대상의 결함이 발견되면 일부만 고쳐 재실행하지 않는다. 해당 60 runs는 QA 증거로 보존하고 MVP 결과는 `Redesign`으로 판정한다. 유효한 60-run 결과는 다음 cycle에서 전부 다시 만든다.

---

## 12. RL 환경으로 승격하는 조건

이 MVP는 RL 환경이 아니다.

다음이 모두 확인된 뒤에만 OpenEnv나 Verifiers 같은 training interface로 변환할지 검토한다.[7][8]

- 두 개 이상의 강한 SUT에서 같은 근본 실패가 반복된다.
- task·gold·grader 오류보다 SUT_ERROR가 충분히 많다.
- grader를 속이는 고득점 저품질 trajectory를 탐지할 수 있다.
- 공개 eval과 겹치지 않는 별도 training task pool을 만들 수 있다.
- 환경을 reset해 같은 초기상태를 재현할 수 있다.
- task 수를 늘려도 전문가 QA가 유지된다.

그때 필요한 추가 작업:

```text
production workflow에서 후보 실패 수집
        ↓
개인정보·권리 필터
        ↓
SME가 task와 gold 작성
        ↓
grader unit test와 reward-hacking test
        ↓
frontier rollout과 transcript triage
        ↓
training pool / sealed eval 분리
        ↓
RL 학습
        ↓
sealed eval로 개선 여부 확인
```

학습용 scalar reward와 외부 평가 scorecard는 분리한다. reward를 만들었다고 평가의 다차원 결과를 하나의 `PASS`로 합치지 않는다.

---

## 13. 법적·커뮤니케이션 한계

「인공지능 및 데이터 기반 행정 활성화에 관한 법률」은 2026-02-27 공포됐고 일반 조항은 2026-08-28 시행된다. 이 문서 기준일에는 아직 일반 시행 전이다.[10] 행정기본법 제20조는 완전히 자동화된 처분을 법률상 근거가 있는 비재량 처분 등에 제한한다.[11]

따라서 MVP는 합성 사실과 공개 법령을 이용한 읽기 전용 평가로 제한한다.

사용 가능한 표현:

> GovAgent-KR Temporal Citation Eval MVP는 특정 동결 corpus에서 기준일별 법령 연혁 선택을 시험하는 개발용 환경이다.

사용하지 않을 표현:

```text
한국 공공 Agent 표준
정부 인증
공공행정 투입 가능 PASS
법적 판단 정확성 보증
우리 회사의 RL 환경 완성
```

20개 item의 결과는 환경 가설을 검증할 뿐 모델·제품·기관 전체의 품질을 대표하지 않는다.

---

## 14. 최종 판단 원칙

이번 MVP의 가장 가치 있는 결과는 높은 점수가 아니다.

- Agent가 왜 틀렸는지 설명할 수 있는 실패
- grader가 왜 맞거나 틀렸는지 재현할 수 있는 판정
- 다음 task를 어디에 만들어야 하는지 알려주는 failure taxonomy

이 세 가지를 10일 안에 얻으면 성공이다.

얻지 못하면 더 큰 benchmark, RL 환경, 표준화와 사업화를 시작하지 않는다.

---

## 참고자료

1. NIA, [AI 에이전트 안전·신뢰성 검증 체계 지원](https://www.nia.or.kr/site/nia_kor/ex/bbs/View.do?bcIdx=29495&cbIdx=78336), 2026-06-08.
2. UK AISI, [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) 및 [Agent Bridge](https://inspect.aisi.org.uk/agent-bridge.html).
3. NIST, [Building Evaluation Probes into Agentic AI](https://www.nist.gov/programs-projects/building-evaluation-probes-agentic-ai), 2026.
4. LBox, [KBL: Korean Legal Benchmark](https://github.com/lbox-kr/kbl), EMNLP 2024.
5. KoBLEX, [Korean Benchmark for Legal Question Answering](https://github.com/daehuikim/KoBLEX), EMNLP 2025.
6. FiscalQA Pro, [Temporal Legal RAG Benchmark](https://github.com/rosecymbler/fiscal-fr-bench), 2026.
7. Meta-PyTorch·Hugging Face, [OpenEnv](https://github.com/meta-pytorch/OpenEnv) 및 [OpenEnv 소개](https://huggingface.co/blog/openenv), 2025.
8. Prime Intellect, [Verifiers: Environments for LLM Reinforcement Learning](https://github.com/PrimeIntellect-ai/verifiers).
9. OpenTelemetry, [GenAI Agent Spans](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md).
10. 국가법령정보센터, [인공지능 및 데이터 기반 행정 활성화에 관한 법률](https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq=283735).
11. 국가법령정보센터, [행정기본법 제20조 관련 조문](https://www.law.go.kr/lsLinkCommonInfo.do?lsJoLnkSeq=1029334663).
