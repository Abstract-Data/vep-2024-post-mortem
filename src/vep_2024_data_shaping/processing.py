from __future__ import annotations
from datetime import datetime, date
from typing import NamedTuple, Generator, Iterable, Optional
import itertools as it

import pandas as pd
import numpy as np
from pathlib import Path
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic import Field as PydanticField, AliasChoices, computed_field, ConfigDict
from icecream import ic

from vep_2024.processing.funcs.loaders.state_loader import StateVoterFile, CreateRecords
from state_voterfiles.utils.pydantic_models.cleanup_model import PreValidationCleanUp

ic.configureOutput(includeContext=False)

PROJECT_FOLDER = Path(__file__).parent.parent.parent.parent
VEP2024_DATA_FOLDER = PROJECT_FOLDER / "vep-2024" / "data"

DOWNLOAD_FOLDER = Path.home() / 'Downloads'
MATCH_FOLDER = VEP2024_DATA_FOLDER / "matches"
VOTERFILE_FOLDER = VEP2024_DATA_FOLDER / "voterfiles"
VEP2024_EXPORTS_FOLDER = VEP2024_DATA_FOLDER / "exports"
TX_TURNOUT_ROSTERS = PROJECT_FOLDER / "texas-turnout-scraper/src/texas_turnout_scraper/tmp/tx_vote_rosters"
SHAPE_EXPORTS = Path(__file__).parents[2] / 'exports'

""" === MATCHED FILES === """
matched_file_list = [
    MATCH_FOLDER / "vep2020_matches.csv",
    MATCH_FOLDER / "vep2022_matches.csv",
    VEP2024_EXPORTS_FOLDER / "20240918_texas_sept24_matches.csv",
    VEP2024_EXPORTS_FOLDER / "20241006_texas_matches.csv",
    VEP2024_EXPORTS_FOLDER / "20241007_texas_matches.csv",
    VEP2024_EXPORTS_FOLDER / "20241206_texas_matches.csv"
]
""" === REFERENCES === """

VUID_REF = 'vuid'
VEP_YEAR_REF = 'vep_year'
WINNER_MARGIN_REF = 'winner_margin'
WINNER_PERCENT_REF = 'winner_percent'
WINNER_PARTY_REF = 'winner_party'
VEP_TOTAL_VOTERS_REF = 'vep total voters'
VEP_PCT_OF_MARGIN_REF = 'vep % of margin'
VEP_WON_RACE_REF = 'vep won race'
OFFICE_DISTRICT_REF = 'office_district'
OFFICE_TYPE_REF = 'office_type'
PARTY_REPUBLICAN = 'R'
COUNTY_REF = 'county'

PARTY_FILTER = PARTY_REPUBLICAN

""" === NAMED TUPLES === """


class DistrictRef(NamedTuple):
    voterfile_ref: str
    result_ref: str


CONGRESSIONAL = DistrictRef(
    voterfile_ref="CONGRESSIONAL district",
    result_ref="CD")

HOUSE_REF = DistrictRef(
    voterfile_ref="legislative lower",
    result_ref="HD")

SENATE_REF = DistrictRef(
    voterfile_ref="legislative upper",
    result_ref="SD")
""" === VALIDATION MODELS ===  """


@pydantic_dataclass
class CriticalData:
    model_config = ConfigDict(str_strip_whitespace=True)
    id: str
    vuid: str = PydanticField(validation_alias=AliasChoices('vuid', 'VUID'))
    cd: int = PydanticField(validation_alias=AliasChoices('congressional district', 'CD'))
    hd: int = PydanticField(validation_alias=AliasChoices('legislative lower', 'HD'))
    sd: int = PydanticField(validation_alias=AliasChoices('legislative upper', 'SD'))
    county: str = PydanticField(validation_alias=AliasChoices('county', 'COUNTY'))
    vep_year: str = PydanticField(validation_alias=AliasChoices('vep_year', 'VEP_YEAR'))
    dob: date = PydanticField(validation_alias=AliasChoices('dob', 'DOB'))
    edr: date = PydanticField(validation_alias=AliasChoices('edr', 'EDR'))
    precinct_name: Optional[str] = PydanticField(default=None, validation_alias=AliasChoices('precinct_name', 'PRECINCT_NAME'))
    precinct_number: str = PydanticField(validation_alias=AliasChoices('precinct_number', 'PRECINCT_NUMBER'))

    @computed_field
    @property
    def age(self) -> int:
        return round(datetime.today().date().year - self.dob.year)

    @computed_field
    @property
    def age_range(self) -> str:
        match self.age:
            case x if x < 18:
                return 'Under 18'
            case x if 18 <= x < 25:
                return '18-24'
            case x if 25 <= x < 35:
                return '25-34'
            case x if 35 <= x < 45:
                return '35-44'
            case x if 45 <= x < 55:
                return '45-54'
            case x if 55 <= x < 65:
                return '55-64'
            case x if 65 <= x < 75:
                return '65-74'
            case x if 75 <= x < 85:
                return '75-84'
            case x if 85 <= x:
                return '85+'
            case _:
                return 'Unknown'


""" === FUNCTIONS === """


def select_only_critical_info(records: Iterable[dict]) -> Generator[CriticalData, None, None]:
    for x in records:
        _x = {}
        _x.update(x['voter_registration'].model_dump(exclude='attributes'))
        _x.update({VEP_YEAR_REF: x[VEP_YEAR_REF]})  # type: ignore
        for d in x['district_set'].districts:
            _x.update({d.name: int(d.number) if d.number else None})
        if x['voter_registration']:
            _x.update({'dob': x['name'].dob})
        try:
            _validate = CriticalData(**_x)
            yield _validate
        except Exception as e:
            BAD_RECORDS.append(e)
            pass
    ic(f"Critical Data filtered with {len(BAD_RECORDS):,} bad records.")


def match_records_at_target_address(records: Iterable[PreValidationCleanUp]) -> Generator[dict, None, None]:
    global SEEN_ADDRESSES, ADDR_DICT, TARGET_COUNT, NON_TARGET_COUNT
    for r in records:
        found, _record = False, None
        if (_vuid := r.voter_registration.vuid) in vuid_dict:
            found = True
            _record = dict(r)
            _record[VEP_YEAR_REF] = vuid_dict.get(_vuid)
            TARGET_COUNT += 1

        for addr in r.address_list:
            if found:
                ADDR_DICT.update({addr.standardized: _record[VEP_YEAR_REF]})
                SEEN_ADDRESSES.add(addr.standardized)
            elif addr.standardized in SEEN_ADDRESSES:
                NON_TARGET_COUNT += 1
                _record = dict(r)
                _record[VEP_YEAR_REF] = ADDR_DICT.get(addr.standardized)

        if _record:
            yield _record

        if TARGET_COUNT > 0 and TARGET_COUNT % 5000 == 0:
            total = TARGET_COUNT + NON_TARGET_COUNT
            ic.enable()
            ic(f"Targets: {TARGET_COUNT:,}. Additional: {NON_TARGET_COUNT:,} ({(NON_TARGET_COUNT / total) * 100:.2f}%). Total: {total:,}")


def office_type_results(office: DistrictRef):
    _results = election_results[election_results[OFFICE_TYPE_REF] == office.result_ref]
    _matches = matched_by_year.groupby(office.voterfile_ref)[excluded_cols].sum().reset_index()
    _turnout = _results.merge(_matches, right_on=office.voterfile_ref, left_on=OFFICE_DISTRICT_REF)
    _turnout = _turnout[_turnout[WINNER_MARGIN_REF] > 0]
    
    _turnout[VEP_PCT_OF_MARGIN_REF] = np.where(
        _turnout[WINNER_PARTY_REF] == PARTY_FILTER,
        round((_turnout[VEP_TOTAL_VOTERS_REF] / _turnout[WINNER_MARGIN_REF]) * 100, 2),
        np.nan
    )
    
    _turnout[VEP_WON_RACE_REF] = np.where(
        (_turnout[WINNER_MARGIN_REF] <= _turnout[VEP_TOTAL_VOTERS_REF])
        &
        (_turnout[WINNER_PARTY_REF] == PARTY_FILTER),
        True,
        ""
    )

    _turnout[OFFICE_DISTRICT_REF] = _turnout[OFFICE_DISTRICT_REF].astype(int)
    
    _turnout.sort_values(
        by=OFFICE_DISTRICT_REF
    ).to_csv(
        DOWNLOAD_FOLDER / f'{datetime.today().strftime("%Y%m%d")}_vep2024_{office.result_ref.lower()}_turnout.csv',
        index=False
    )
    return _turnout


ADDR_DICT = dict()
SEEN_ADDRESSES = set()
BAD_RECORDS = []
TARGET_COUNT = 0
NON_TARGET_COUNT = 0

texas_vep_start_date = datetime.strptime('10/20/2023', '%m/%d/%Y').date()

matches_2020 = pd.read_csv(MATCH_FOLDER / "vep2020_matches.csv")[VUID_REF.upper()].map(str).to_list()
matches_2022 = pd.read_csv(MATCH_FOLDER / "vep2022_matches.csv")[VUID_REF.upper()].map(str).to_list()
latest_matches_df1 = pd.read_csv(VEP2024_EXPORTS_FOLDER / "20240918_texas_sept24_matches.csv")
latest_matches_df2 = pd.read_csv(VEP2024_EXPORTS_FOLDER / "20241006_texas_matches.csv")
latest_matches_df3 = pd.read_csv(VEP2024_EXPORTS_FOLDER / "20241007_texas_matches.csv")
latest_matches_df4 = pd.read_csv(VEP2024_EXPORTS_FOLDER / "20241206_texas_matches.csv")
latest_matches_df = pd.concat([latest_matches_df1, latest_matches_df2, latest_matches_df3, latest_matches_df4])
latest_matches_df = latest_matches_df.drop_duplicates(subset=[VUID_REF])
latest_matches_vuids = [(str(y), 'vep2024') for y in latest_matches_df[VUID_REF].to_list()]
latest_matches_vuids.extend([(x, 'vep2020') for x in matches_2020])
latest_matches_vuids.extend([(x, 'vep2022') for x in matches_2022])
all_matched_vuids = pd.DataFrame(list(set(latest_matches_vuids)), columns=[VUID_REF.upper(), VEP_YEAR_REF.upper()])

last_month_texas_voterfile = StateVoterFile(state='texas')
last_month = last_month_texas_voterfile.read(VOTERFILE_FOLDER / 'texas/texasnovember2024.csv')
last_month_texas_voterfile.validate(x for x in last_month if int(x['EDR']) >= 20231001)

vuid_dict = all_matched_vuids.set_index(VUID_REF.upper())[VEP_YEAR_REF.upper()].to_dict()

RECORD_CREATOR = CreateRecords()
validated_records = RECORD_CREATOR.create_records(last_month_texas_voterfile.validation.valid)
record_set = match_records_at_target_address(validated_records)
critical_data = select_only_critical_info(record_set)

turnout_list_path = TX_TURNOUT_ROSTERS / "2024/2024 NOVEMBER 5TH GENERAL ELECTION/2024-12-04-TX-2024-GE.csv"
turnout_list = pd.read_csv(turnout_list_path)
turnout_vuids = dict.fromkeys(str(x) for x in turnout_list['voter_id'].to_list())

matched_that_voted = (x for x in critical_data if x.vuid in turnout_vuids)
matched_did_not_vote = (x for x in critical_data if x.vuid not in turnout_vuids)


matched_that_voted_df = pd.DataFrame(matched_that_voted)
matched_that_voted_df.to_csv(SHAPE_EXPORTS / '2024_texas_matched_voted.csv', index=False)
matched_by_year = pd.crosstab(
    index=[
        matched_that_voted_df[COUNTY_REF],
        matched_that_voted_df[SENATE_REF.result_ref.lower()],
        matched_that_voted_df[HOUSE_REF.result_ref.lower()],
        matched_that_voted_df[CONGRESSIONAL.result_ref.lower()]
    ],
    columns=matched_that_voted_df[VEP_YEAR_REF],
    values=matched_that_voted_df[VUID_REF],
    aggfunc='count',
    margins=True,
    margins_name=VEP_TOTAL_VOTERS_REF
).reset_index().fillna(0).replace("", pd.NA).map(lambda x: int(x) if isinstance(x, float) else x)

excluded_cols = [
    x for x in matched_by_year.columns if x != COUNTY_REF
]

matched_by_year[excluded_cols] = matched_by_year[excluded_cols].astype(int)

election_results_path = DOWNLOAD_FOLDER / '20241111_election_results.csv'
election_results = pd.read_csv(election_results_path)
election_results[WINNER_PERCENT_REF] = round(election_results[WINNER_PERCENT_REF] * 100, 2)

cd_turnout = office_type_results(CONGRESSIONAL)
hd_turnout = office_type_results(HOUSE_REF)
sd_turnout = office_type_results(SENATE_REF)
