"""命令行演示：走一遍完整的 LangGraph 流程，打印每个节点的产出。

python -m app.demo
"""

import argparse
import json
from pathlib import Path

from app.config import BASE_DIR
from app.graph import run_agent

LEGACY_ASSETS = BASE_DIR.parent / "healthy_diet_recommender" / "static" / "用于测试的文件"


def main() -> None:
    parser = argparse.ArgumentParser(description="健康膳食 Agent 演示")
    parser.add_argument("--query", default="根据我的体检报告和冰箱里的食材，推荐一份控糖晚餐")
    parser.add_argument("--report", default=None, help="体检报告图片/PDF 路径")
    parser.add_argument("--fridge", default=None, help="冰箱照片路径")
    parser.add_argument("--user-id", type=int, default=1)
    args = parser.parse_args()

    report = args.report
    fridge = args.fridge
    if report is None and (LEGACY_ASSETS / "tjbg_3.png").exists():
        report = str(LEGACY_ASSETS / "tjbg_3.png")
    if fridge is None and (LEGACY_ASSETS / "bx_1.jpg").exists():
        fridge = str(LEGACY_ASSETS / "bx_1.jpg")

    result = run_agent(
        query=args.query,
        user_id=args.user_id,
        report_image_path=report,
        fridge_image_path=fridge,
    )

    print("\n=== 1. 意图 ===")
    print(result.get("intent"))

    print("\n=== 2. 健康画像 ===")
    print(json.dumps(result.get("health_metrics", {}), ensure_ascii=False, indent=2))
    print("约束:", result.get("profile", {}).get("constraints"))

    print("\n=== 3. 识别到的食材 ===")
    print(result.get("ingredients"))

    print("\n=== 4. 知识库检索（混合检索） ===")
    for hit in result.get("retrieved_knowledge", []):
        print(f"- [{hit.get('score')}] {hit.get('text')[:60]}...")

    print("\n=== 5. 工具执行轨迹 ===")
    for trace in result.get("tool_trace", []):
        status = "ok" if trace["ok"] else f"fail({trace['error']})"
        print(f"- {trace['tool']}: {status}")

    print("\n=== 6. 反思与修订 ===")
    print(f"修订次数: {result.get('revision_count')}  评分: {result.get('reflection', {}).get('score')}")
    print("问题:", result.get("reflection", {}).get("issues"))

    print("\n=== 7. 最终输出 ===\n")
    print(result.get("final_answer"))

    if result.get("errors"):
        print("\n=== 错误 ===\n", result["errors"])


if __name__ == "__main__":
    Path(BASE_DIR / "data").mkdir(exist_ok=True)
    main()
