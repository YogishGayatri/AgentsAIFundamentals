from typing import Literal
from pydantic import BaseModel, Field, ValidationError

class Part(BaseModel):
    name: str
    quantity: int = Field(ge=1, description="must be at least 1")
    material: Literal["aluminium", "steel", "titanium"]   # only these allowed

p = Part(name="bracket", quantity=10, material="aluminium")
print(p.material)          
print(p.model_dump())      

# Pydantic validtion
try:
    Part(name="bracket", quantity=0, material="wood")
except ValidationError as e:
    for err in e.errors():
        print(err["loc"], "->", err["msg"])
print(Part(name="x", quantity="10", material="steel").quantity)   # 10  (an int)