"""核心逻辑测试：指标抽取、约束推导、评审打分、整图运行。"""

import pytest

from app.graph import build_graph, run_agent
from app.nodes.collect import derive_constraints, extract_metrics
from app.nodes.reflect import evaluate_recipe

HYPERTENSION_REPORT = (
    "体检报告 姓名:赵建国 身高168cm 体重74kg BMI26.2 "
    "空腹血糖6.1mmol/L 总胆固醇5.5mmol/L 血压152/94mmHg 过敏史:无"
)

DIABETES_REPORT = (
    "体检报告 BMI26.7 空腹血糖6.4mmol/L 糖化血红蛋白6.3% "
    "总胆固醇5.1mmol/L 血压126/82mmHg 过敏史:花生、海鲜"
)


def test_extract_metrics_and_constraints():
    metrics = extract_metrics(HYPERTENSION_REPORT)
    assert metrics["bmi"] == 26.2
    assert metrics["systolic"] == 152
    assert metrics["diastolic"] == 94

    constraints = derive_constraints(metrics)
    assert "低盐" in constraints
    assert "控热量" in constraints


def test_allergy_constraint():
    constraints = derive_constraints(extract_metrics(DIABETES_REPORT))
    assert "控糖" in constraints
    assert "避敏:花生" in constraints
    assert "避敏:海鲜" in constraints


def test_reflector_rejects_shallow_recipe():
    shallow = {
        "ingredients": ["西红柿", "鸡蛋"],
        "steps": ["切块", "翻炒"],
        "nutrition_note": "",
        "constraint_notes": [],
        "est_kcal": 560,
    }
    result = evaluate_recipe(shallow, ["西红柿", "鸡蛋", "青椒"], ["低脂", "控热量"])
    assert result["passed"] is False
    assert "未体现健康约束「低脂」" in result["issues"]


def test_reflector_accepts_complete_recipe():
    complete = {
        "ingredients": ["西红柿", "鸡蛋"],
        "steps": ["处理食材", "热锅少油", "翻炒调味", "出锅装盘"],
        "nutrition_note": "蛋白质充足",
        "constraint_notes": ["已按「低脂」调整", "已按「控热量」调整"],
        "est_kcal": 420,
    }
    result = evaluate_recipe(complete, ["西红柿", "鸡蛋"], ["低脂", "控热量"])
    assert result["passed"] is True
    assert result["score"] >= 80


@pytest.mark.parametrize("report", [HYPERTENSION_REPORT, DIABETES_REPORT])
def test_graph_end_to_end_with_revision(report, tmp_path, monkeypatch):
    """多模态采集 → 规划 → 执行 → 反思不通过 → 回到规划 → 通过收尾。

    测试里强制 mock 掉所有外部依赖（OCR / YOLO / LLM），
    避免受 .env 真实配置影响（比如 LLM_MOCK=0 时 DeepSeek 第一次就出完美菜谱）。
    """
    report_path = tmp_path / "report.png"
    report_path.write_text("")

    import app.adapters.vision as vision  # noqa: F401 - 保证模块已加载
    import app.nodes.collect as collect_mod
    import app.tools.implementations as impl

    # 注意：implementations 在 import 时就绑定了函数引用，必须打在 impl 上才生效
    monkeypatch.setattr(impl, "parse_health_report", lambda _: (report, "mock"))
    monkeypatch.setattr(
        impl,
        "detect_ingredients",
        lambda _: {"source": "mock", "ingredients": [{"name": "西红柿", "confidence": 0.9}, {"name": "鸡蛋", "confidence": 0.9}], "fallback": None},
    )
    # 关掉真实 LLM，保证首轮一定不达标、第二轮修订后通过
    monkeypatch.setattr(impl._llm, "_client", None)
    monkeypatch.setattr(collect_mod._llm, "_client", None)

    graph = build_graph()
    result = run_agent(
        query="推荐一份适合我的晚餐",
        report_image_path=str(report_path),
        fridge_image_path=str(tmp_path / "fridge.jpg"),
        graph=graph,
    )

    assert result["intent"] == "generate_recipe"
    assert result["revision_count"] >= 2          # 至少触发一轮反思修订
    assert result["reflection"]["passed"] is True
    assert result["draft_recipe"]["title"]
    assert "推荐菜谱" in result["final_answer"]
    assert result["errors"] == []
