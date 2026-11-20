"""节点 6/7：Reflector（质量评审 + 迭代） 与 Finalize（收尾输出）。

Reflector 是这张图区别于「一条 Prompt 直出」的关键：
它用可解释的评分规则判断菜谱是否真的满足用户约束，
不达标就带着具体问题回到 Planner 再来一轮。
"""

import re
from typing import Any

from app.config import settings
from app.state import DietState

# 模型输出常自带「1.」「-」前缀，收尾时统一剥掉再重新编号
_STEP_NOISE = re.compile(r"^\s*(?:\d+[.、)．]|[-*])\s*")

_METRIC_LABELS = {
    "bmi": "BMI",
    "height_cm": "身高cm",
    "weight_kg": "体重kg",
    "fasting_glucose": "空腹血糖",
    "hba1c": "糖化血红蛋白",
    "total_cholesterol": "总胆固醇",
    "triglyceride": "甘油三酯",
    "ldl": "低密度脂蛋白",
    "systolic": "收缩压",
    "diastolic": "舒张压",
}


def evaluate_recipe(
    recipe: dict[str, Any],
    available_ingredients: list[str],
    constraints: list[str],
) -> dict[str, Any]:
    """100 分制评分，返回分数 / 是否通过 / 问题清单。"""
    score = 0
    issues: list[str] = []

    # 1. 食材利用率（20）
    used = set(recipe.get("ingredients", [])) & set(available_ingredients)
    if len(used) >= 2:
        score += 20
    elif len(used) == 1:
        score += 10
        issues.append("仅使用了 1 种现有食材，建议多用一种")
    else:
        issues.append("菜谱没有使用冰箱现有食材")

    # 2. 步骤完整性（20）
    steps = len(recipe.get("steps", []))
    if steps >= 3:
        score += 20
    elif steps == 2:
        score += 10
        issues.append("烹饪步骤过于简略，缺少火候/调味说明")
    else:
        issues.append("缺少可执行的烹饪步骤")

    # 3. 健康约束覆盖（30）
    notes = "、".join(recipe.get("constraint_notes", []))
    if constraints:
        per = 30 / len(constraints) if constraints else 0
        for constraint in constraints:
            if constraint in notes or constraint.split(":")[0] in notes:
                score += per
            else:
                issues.append(f"未体现健康约束「{constraint}」")
    else:
        score += 30

    # 4. 营养说明（15）
    if recipe.get("nutrition_note"):
        score += 15
    else:
        issues.append("缺少营养说明")

    # 5. 单餐热量合理性（15）
    kcal = recipe.get("est_kcal") or 0
    if 300 <= kcal <= 700:
        score += 15
    elif kcal:
        issues.append(f"单餐热量 {kcal}kcal 偏离 300-700 区间")

    score = int(round(score))
    return {
        "score": score,
        "passed": score >= settings.min_quality_score,
        "issues": issues,
        "hint": "；".join(issues),
    }


def reflect_node(state: DietState) -> dict[str, Any]:
    recipe = state.get("draft_recipe") or {}
    profile = state.get("profile", {})
    constraints = profile.get("constraints", []) or []

    if not recipe:
        return {
            "reflection": {"score": 0, "passed": False, "issues": ["未生成菜谱"], "hint": "重新生成菜谱"},
            "revision_count": state.get("revision_count", 0) + 1,
        }

    result = evaluate_recipe(recipe, state.get("ingredients", []) or [], constraints)
    return {"reflection": result, "revision_count": state.get("revision_count", 0) + 1}


def route_after_reflect(state: DietState) -> str:
    reflection = state.get("reflection", {}) or {}
    revisions = state.get("revision_count", 0)
    if not reflection.get("passed") and revisions < settings.max_revisions:
        return "plan"
    return "finalize"


def finalize_node(state: DietState) -> dict[str, Any]:
    recipe = state.get("draft_recipe") or {}
    nutrition = state.get("nutrition_report", {}) or {}
    reflection = state.get("reflection", {}) or {}
    metrics = state.get("health_metrics", {}) or {}
    profile = state.get("profile", {}) or {}

    if not recipe:
        knowledge = state.get("retrieved_knowledge", []) or []
        answer = "未能生成菜谱。"
        if knowledge:
            answer += "\n\n可参考的膳食建议：\n" + "\n".join(f"- {k['text']}" for k in knowledge[:2])
        return {"final_answer": answer}

    lines = [f"## 推荐菜谱：{recipe.get('title', '家常菜')}"]
    if profile.get("constraints"):
        lines.append(f"**健康约束**：{'、'.join(profile['constraints'])}")
    if metrics:
        brief = "、".join(f"{_METRIC_LABELS.get(k, k)}={v}" for k, v in metrics.items() if k != "allergy")
        lines.append(f"**体检指标**：{brief}")

    lines.append("\n**食材**：" + "、".join(recipe.get("ingredients", [])))
    lines.append("\n**烹饪步骤**")
    steps = [_STEP_NOISE.sub("", s) for s in recipe.get("steps", [])]
    lines.extend(f"{i}. {step}" for i, step in enumerate(steps, 1))

    if recipe.get("nutrition_note"):
        lines.append(f"\n**营养说明**：{recipe['nutrition_note']}")
    if recipe.get("constraint_notes"):
        lines.append("**约束适配**：" + "、".join(recipe["constraint_notes"]))

    if nutrition.get("total"):
        total = nutrition["total"]
        lines.append(
            f"\n**营养估算**：热量 {total['kcal']}kcal / 蛋白质 {total['protein_g']}g / "
            f"脂肪 {total['fat_g']}g / 碳水 {total['carb_g']}g / 钠 {total['sodium_mg']}mg"
        )

    lines.append(
        f"\n> 质量评分 {reflection.get('score', 0)}/100，"
        f"修订 {max(state.get('revision_count', 0) - 1, 0)} 次"
    )

    return {"final_answer": "\n".join(lines)}
