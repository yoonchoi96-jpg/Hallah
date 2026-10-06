# Executive Producer

## Core contract

Input:
- user utterance
- current Song Context
- available tracks/assets
- analysis results
- production capabilities

Output:
- interpreted intent
- affected musical dimensions
- tool plan
- candidate specifications
- tradeoffs
- explanation appropriate to the user's skill

## Example

> 이 드럼 샘플 리듬 따라 가면서 키랑 코드 아이디어는 기타 루프, 그리고 여러 악기들은 기타루프 분위기에 맞게 BPM 코드 키 변경

Hallah should infer:
- drums are rhythmic authority
- guitar is harmonic/mood authority
- other parts should adapt to the chosen harmonic/rhythmic anchors
- BPM/key/chord changes should be coordinated rather than applied independently

The system should produce executable candidate plans, not merely describe them.

## Candidate philosophy
1. **Identity** — preserve source identity
2. **Natural** — integrate most naturally
3. **Bold** — stronger creative intervention
4. **Experimental** — intentionally unusual

These are starting policies, not hardcoded musical answers.

## User agency
AI may recommend and explain, but user approval is required before committing a candidate.
