"""Strict local validation of teacher examples and generation settings."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from task2_genai.policies import POLICY_BY_TOPIC

TEACHER_MODEL = "openai/gpt-oss-120b"
STUDENT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
RiskLevel = Literal["low", "medium", "high"]


class PolicyExample(BaseModel):
    """Validated record; ID uniqueness is enforced across the dataset."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)
    id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    topic: str = Field(min_length=1, max_length=80)
    policy_excerpt: str = Field(min_length=30, max_length=1600)
    scenario: str = Field(min_length=20, max_length=2000)
    category: str = Field(min_length=1, max_length=80)
    risk_level: RiskLevel
    recommended_action: str = Field(min_length=5, max_length=400)
    rationale: str = Field(min_length=15, max_length=1000)

    @model_validator(mode="after")
    def check_category(self) -> "PolicyExample":
        policy = POLICY_BY_TOPIC.get(self.topic)
        if policy is None or policy.category != self.category:
            raise ValueError("topic/category must match the internal taxonomy")
        return self


class GenerationConfig(BaseModel):
    """160 candidates in batches of five by default; no model training."""

    model_config = ConfigDict(extra="forbid")
    candidate_count: int = Field(default=160, ge=120, le=1000)
    batch_size: int = Field(default=5, ge=5, le=20)
    minimum_clean: int = Field(default=120, ge=120)
    target_clean: int = Field(default=130, ge=120)
    seed: int = 42

    @model_validator(mode="after")
    def check_size(self) -> "GenerationConfig":
        if self.minimum_clean > self.candidate_count:
            raise ValueError("minimum_clean cannot exceed candidate_count")
        if not self.minimum_clean <= self.target_clean <= self.candidate_count:
            raise ValueError(
                "target_clean must be between minimum_clean and candidate_count"
            )
        return self
