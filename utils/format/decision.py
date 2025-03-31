from pydantic import BaseModel, Field

class OptimizationDecision(BaseModel):
    function_id: str = Field(..., title="Function ID", description="ID of the strategy to optimize next (F1, F2, or F3)")
    rationale: str = Field(..., title="Rationale", description="Reasoning behind selecting this strategy for the next optimization step")
    suggestions: str = Field(..., title="Suggestions", description="Specific suggestions for improving this strategy based on its current state")