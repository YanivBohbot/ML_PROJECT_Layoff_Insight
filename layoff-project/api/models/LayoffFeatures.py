from pydantic import BaseModel


class LayoffFeatures(BaseModel):
    total_laid_off: float
    perc_laid_off: float
    funds_raised: str
    industry: str
    country: str
    stage: str
