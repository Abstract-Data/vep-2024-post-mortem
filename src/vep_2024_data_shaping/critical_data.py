from __future__ import annotations
from datetime import datetime, date
from typing import Optional

from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic import Field as PydanticField, AliasChoices, model_validator, ConfigDict, field_validator
from references import *

def UPPER_AND_LOWER(val: str | tuple) -> list[str]:
    """
    Converts a string or a tuple of strings to a list containing both upper and lower case versions.

    Args:
        val (str | tuple): A string or a tuple of strings to be converted.

    Returns:
        list[str]: A list containing the upper and lower case versions of the input string(s).
    """
    if isinstance(val, str):
        return [val.upper(), val.lower()]
    return [item for sublist in val for item in (sublist.upper(), sublist.lower())]


@pydantic_dataclass
class CriticalData:
    model_config = ConfigDict(str_strip_whitespace=True)
    id: str
    vuid: int = PydanticField(
        validation_alias=AliasChoices(*UPPER_AND_LOWER(VUID_REF)))
    cd: int = PydanticField(
        validation_alias=AliasChoices(*UPPER_AND_LOWER(CONGRESSIONAL)))
    hd: int = PydanticField(
        validation_alias=AliasChoices(*UPPER_AND_LOWER(HOUSE_REF)))
    sd: int = PydanticField(
        validation_alias=AliasChoices(*UPPER_AND_LOWER(SENATE_REF)))
    county: str = PydanticField(
        validation_alias=AliasChoices(*UPPER_AND_LOWER(COUNTY_REF)))
    vep_year: str = PydanticField(
        validation_alias=AliasChoices(*UPPER_AND_LOWER(VEP_YEAR_REF)))
    dob: date = PydanticField(
        validation_alias=AliasChoices(*UPPER_AND_LOWER(DOB_REF)))
    age: Optional[int] = PydanticField(default=None)
    age_range: Optional[str] = PydanticField(default=None)
    date_voted: Optional[date] = PydanticField(
        default=None,
        validation_alias=AliasChoices(*UPPER_AND_LOWER(DATE_VOTED_REF)))
    edr: date = PydanticField(
        validation_alias=AliasChoices(*UPPER_AND_LOWER(REGISTRATION_DATE_REF)))
    precinct_name: Optional[str] = PydanticField(
        default=None,
        validation_alias=AliasChoices(*UPPER_AND_LOWER(PRECINCT_NAME_REF))
    )
    precinct_number: str = PydanticField(
        validation_alias=AliasChoices(*UPPER_AND_LOWER(PRECINCT_NUMBER_REF))
    )
    voted_in_2020: Optional[bool] = PydanticField(default=None)
    voted_in_2022: Optional[bool] = PydanticField(default=None)
    voted_in_2024: Optional[bool] = PydanticField(default=None)

    @field_validator('date_voted')
    def validate_date_voted(cls, v):
        if v is not None and not isinstance(v, date):
            return datetime.strptime(v, '%Y-%m-%d').date()
        return v

    @model_validator(mode='after')
    def calculate_age(self) -> CriticalData:
        self.age = round(datetime.today().date().year - self.dob.year)
        return self

    @model_validator(mode='after')
    def filter_age_range(self) -> CriticalData:
        match self.age:
            case x if x < 18:
                _r = 'Under 18'
            case x if 18 <= x < 25:
                _r = '18-24'
            case x if 25 <= x < 35:
                _r =  '25-34'
            case x if 35 <= x < 45:
                _r = '35-44'
            case x if 45 <= x < 55:
                _r = '45-54'
            case x if 55 <= x < 65:
                _r = '55-64'
            case x if 65 <= x < 75:
                _r = '65-74'
            case x if 75 <= x < 85:
                _r = '75-84'
            case x if 85 <= x:
                _r = '85+'
            case _:
                _r = 'Unknown'
        self.age_range = _r
        return self
