from pydantic import BaseModel, Field

class Code(BaseModel):
    code: str = Field(..., title="Code snippet", description="Python code snippet")
    explain: str = Field(..., title="Explanation", description="Explanation of the code snippet")