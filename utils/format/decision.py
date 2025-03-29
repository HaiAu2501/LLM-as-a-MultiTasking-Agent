from pydantic import BaseModel, Field

class OptimizationDecision(BaseModel):
    function_id: str = Field(..., title="Function ID", description="ID of the function to optimize next (F1, F2, or F3)")
    rationale: str = Field(..., title="Rationale", description="Reasoning behind selecting this function for the next optimization step")
    suggestions: str = Field(..., title="Suggestions", description="Specific suggestions for improving this function based on its current state")