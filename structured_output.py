from pydantic import BaseModel


class MathAnswer(BaseModel):
    answer: float
    unit: str
    formula: str
