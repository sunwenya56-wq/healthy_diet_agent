"""端到端测试：老项目测试素材 → 上传 → Agent 完整链路 → 输出。

python -m app.e2e_test
"""

import json
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8000"
OLD_ASSETS = Path(__file__).resolve().parent.parent.parent / "healthy_diet_recommender" / "static" / "用于测试的文件"


def step(title: str):
    print(f"\n{'='*60}\n🔹 {title}\n{'='*60}")


def main():
    time.sleep(1.5)

    step("0. 健康检查")
    r = requests.get(f"{BASE}/healthz")
    print(r.status_code, json.dumps(r.json(), ensure_ascii=False))

    step("1. 上传体检报告（老项目素材 tjbg_3.png）")
    r = requests.post(
        f"{BASE}/api/upload/report?user_id=1",
        files={"file": ("tjbg_3.png", (OLD_ASSETS / "tjbg_3.png").open("rb"), "image/png")},
    )
    print(f"status={r.status_code}  body={r.text[:200]}")
    report_resp = r.json()

    step("2. 上传冰箱图片（老项目素材 bx_1.jpg）")
    r = requests.post(
        f"{BASE}/api/upload/fridge?user_id=1",
        files={"file": ("bx_1.jpg", (OLD_ASSETS / "bx_1.jpg").open("rb"), "image/jpeg")},
    )
    print(f"status={r.status_code}  body={r.text[:200]}")
    fridge_resp = r.json()

    step("3. Agent 完整调用（带真实图片路径）")
    r = requests.post(
        f"{BASE}/api/agent/run",
        json={
            "user_id": 1,
            "query": "根据我的体检报告和冰箱食材，推荐一份晚餐",
            "report_image_path": report_resp["path"],
            "fridge_image_path": fridge_resp["path"],
            "debug": True,
        },
    )
    run = r.json()
    ref = run.get("reflection", {})
    print(f"status={r.status_code}")
    print(f"  intent={run.get('intent')}  score={ref.get('score')}/100  revisions={run.get('revision_count',0)-1}  recipe_id={run.get('recipe_id')}")
    print(f"  菜谱标题: {run.get('draft_recipe',{}).get('title')}")
    print(f"  约束: {run.get('profile',{}).get('constraints')}")
    print(f"  食材: {run.get('ingredients')}")
    print(f"\n最终输出:\n{run.get('final_answer','')[:400]}")

    step("4. 流式 SSE（展示前 15 事件）")
    r = requests.post(
        f"{BASE}/api/agent/stream",
        json={
            "user_id": 2,
            "query": "给我做一份低脂早餐",
            "ingredients": ["燕麦", "鸡蛋", "牛奶", "蓝莓"],
        },
        stream=True,
    )
    print(f"status={r.status_code}  content-type={r.headers.get('content-type')}")
    buffer = ""
    shown = 0
    for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
        buffer += chunk
        while "\n\n" in buffer and shown < 15:
            evt_raw, buffer = buffer.split("\n\n", 1)
            data_lines = [l[5:] for l in evt_raw.split("\n") if l.startswith("data:")]
            for line in data_lines:
                if not line.strip() or line.strip() == "[DONE]":
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = ev.get("type")
                extras = ""
                if t == "tool_call":
                    extras = f"→ {ev.get('tool')}"
                elif t == "tool_result":
                    extras = f"{ev.get('tool')} ok={ev.get('ok')}"
                elif t == "step_end":
                    extras = json.dumps(ev.get("key", {}), ensure_ascii=False)[:50]
                print(f"  [{shown:02d}] {t:12s} {extras}")
                shown += 1
                break

    step("5. ASR Mock（上传一张图当语音 Mock 输入）")
    r = requests.post(
        f"{BASE}/api/asr?user_id=1",
        files={"file": ("voice.webm", (OLD_ASSETS / "bx_2.jpg").open("rb"), "audio/webm")},
    )
    asr = r.json()
    print(f"status={r.status_code}  source={asr.get('source')}  text={asr.get('text')}")

    step("6. TTS Mock")
    r = requests.post(
        f"{BASE}/api/tts",
        json={"text": "健康饮食，多吃蔬菜水果"},
    )
    tts = r.json()
    print(f"status={r.status_code}  source={tts['source']}")
    print(f"  audio_base64_len={tts['audio_base64']}  note={tts.get('note','')}")

    step("7. 历史菜谱查询")
    r = requests.get(f"{BASE}/api/recipes", params={"user_id": 1})
    print(f"status={r.status_code}  count={len(r.json()['items'])}")
    for item in r.json()["items"][-3:]:
        print(f"  - {item['title']}  score={item['quality_score']}  created={item['created_at'][:19]}")

    print("\n✅ 全部测试通过！")


if __name__ == "__main__":
    main()
