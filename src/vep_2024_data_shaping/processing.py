from __future__ import annotations
from datetime import datetime, date
from typing import NamedTuple, Generator, Iterable, Optional, ForwardRef
from functools import partial

import pandas as pd
import numpy as np
from pathlib import Path
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic import Field as PydanticField, AliasChoices, model_validator, ConfigDict
from icecream import ic
from tqdm import tqdm

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
VEP_PREVIOUS_YEARS_MATCHED_VOTERS = [
    MATCH_FOLDER / "vep2020_matches.csv",
    MATCH_FOLDER / "vep2022_matches.csv",
]
VEP2024_MATCHED_VOTERS = [
    VEP2024_EXPORTS_FOLDER / "20240918_texas_sept24_matches.csv",
    VEP2024_EXPORTS_FOLDER / "20241006_texas_matches.csv",
    VEP2024_EXPORTS_FOLDER / "20241007_texas_matches.csv",
    VEP2024_EXPORTS_FOLDER / "20241206_texas_matches.csv"
]

TURNOUT_LIST_PATH = TX_TURNOUT_ROSTERS / "2024/2024 NOVEMBER 5TH GENERAL ELECTION/2024-12-04-TX-2024-GE.csv"
ELECTION_RESULTS_PATH = DOWNLOAD_FOLDER / '20241111_election_results.csv'
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
DOB_REF = 'dob'
REGISTRATION_DATE_REF = 'edr'
PRECINCT_NAME_REF = 'precinct_name'
PRECINCT_NUMBER_REF = 'precinct_number'
DATE_VOTED_REF = 'date_voted'

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
    vuid: int = PydanticField(
        validation_alias=AliasChoices(VUID_REF, VUID_REF.upper()))
    cd: int = PydanticField(
        validation_alias=AliasChoices(CONGRESSIONAL.result_ref, CONGRESSIONAL.voterfile_ref))
    hd: int = PydanticField(
        validation_alias=AliasChoices(HOUSE_REF.result_ref, HOUSE_REF.voterfile_ref))
    sd: int = PydanticField(
        validation_alias=AliasChoices(SENATE_REF.result_ref, SENATE_REF.voterfile_ref))
    county: str = PydanticField(
        validation_alias=AliasChoices(COUNTY_REF, COUNTY_REF.upper()))
    vep_year: str = PydanticField(
        validation_alias=AliasChoices(VEP_YEAR_REF, VEP_YEAR_REF.upper()))
    dob: date = PydanticField(
        validation_alias=AliasChoices(DOB_REF, DOB_REF.upper()))
    age: Optional[int] = PydanticField(default=None)
    age_range: Optional[str] = PydanticField(default=None)
    date_voted: date = PydanticField(
        validation_alias=AliasChoices(DATE_VOTED_REF, DATE_VOTED_REF.upper()))
    edr: date = PydanticField(
        validation_alias=AliasChoices(REGISTRATION_DATE_REF, REGISTRATION_DATE_REF.upper()))
    precinct_name: Optional[str] = PydanticField(
        default=None,
        validation_alias=AliasChoices(PRECINCT_NAME_REF, PRECINCT_NAME_REF.upper())
    )
    precinct_number: str = PydanticField(
        validation_alias=AliasChoices(PRECINCT_NUMBER_REF, PRECINCT_NUMBER_REF.upper())
    )

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


""" === FUNCTIONS === """


def select_only_critical_info(
        records: Iterable[ForwardRef('RecordBaseModel')],
        turnout_dict: dict[str, date]) -> Generator[CriticalData, None, None]:
    ic("Filtering critical data.")
    for x in records:
        _x = {}
        _x.update(x['voter_registration'].model_dump(exclude='attributes'))
        _x.update({VEP_YEAR_REF: x[VEP_YEAR_REF]})  # type: ignore
        _x.update({'date_voted': turnout_dict.get(x['voter_registration'].vuid)})
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


def match_records_at_target_address(
        records: Iterable[PreValidationCleanUp | ForwardRef('RecordBaseModel')]
) -> Generator[dict, None, None]:
    global SEEN_ADDRESSES, ADDR_DICT, TARGET_COUNT, NON_TARGET_COUNT
    for r in tqdm(records, desc="Matching target records and voters at target address."):
        found, _record = False, None
        if (_vuid := int(r.voter_registration.vuid)) in vuid_dict:
            found = True
            _record = dict(r)
            _record[VEP_YEAR_REF] = vuid_dict.get(_vuid)
            TARGET_COUNT += 1

        for addr in r.address_list:
            if found:
                ADDR_DICT.update({addr.standardized: _record[VEP_YEAR_REF]})
                SEEN_ADDRESSES.add(addr.standardized)
                SEEN_ADDRESS_EDR.update({addr.standardized: r.voter_registration.edr})
            elif addr.standardized in SEEN_ADDRESSES:
                if r.voter_registration.edr >= SEEN_ADDRESS_EDR.get(addr.standardized):
                    NON_TARGET_COUNT += 1
                    _record = dict(r)
                    _record[VEP_YEAR_REF] = ADDR_DICT.get(addr.standardized)
                    break

        if _record:
            yield _record

        if TARGET_COUNT > 0 and TARGET_COUNT % 5000 == 0:
            total = TARGET_COUNT + NON_TARGET_COUNT
            ic.enable()
            ic(f"Targets: {TARGET_COUNT:,}. Additional: {NON_TARGET_COUNT:,} ({(NON_TARGET_COUNT / total) * 100:.2f}%). Total: {total:,}")


def office_type_results(office: DistrictRef, election_results_df: pd. DataFrame):
    ic("Creating turnout data for ", office.result_ref)
    _results = election_results_df[election_results_df[OFFICE_TYPE_REF] == office.result_ref]
    _results[OFFICE_DISTRICT_REF] = _results[OFFICE_DISTRICT_REF].astype(int)
    _matches = matched_by_year.groupby(office.result_ref.lower())[excluded_cols].sum().reset_index()
    _turnout = _results.merge(_matches, right_on=office.result_ref.lower(), left_on=OFFICE_DISTRICT_REF)
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
        DOWNLOAD_FOLDER / f'{datetime.today().strftime("%Y%m%d")}_vep2024_{office.result_ref.lower()}_turnout1.csv',
        index=False
    )
    return _turnout


def get_previous_vep_year_vuids(rec_ls: list = None) -> tuple[str, str]:
    ic("Getting previous years VEP matched voters.")
    if not rec_ls:
        rec_ls = VEP_PREVIOUS_YEARS_MATCHED_VOTERS
    for f in rec_ls:
        _data = pd.read_csv(f)[VUID_REF.upper()].map(str).to_list()
        for x in _data:
            yield x, f.stem.split('_')[0]


def get_current_year_dataframe(rec_ls: list = None) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    if not rec_ls:
        rec_ls = VEP2024_MATCHED_VOTERS

    ic("Getting current year VEP matched voters.")
    _data = pd.concat([pd.read_csv(f) for f in rec_ls])
    data = _data.drop_duplicates(subset=[VUID_REF])
    vuid_list = [(str(y), 'vep2024') for y in data[VUID_REF].to_list()]
    return data, vuid_list


def get_turnout_data(turnout_file: Path = TURNOUT_LIST_PATH) -> tuple[pd.DataFrame, dict, dict]:
    ic("Getting turnout data.")
    _turnout_list = pd.read_csv(turnout_file)
    _turnout_vuids = dict.fromkeys(int(x) for x in _turnout_list['voter_id'].to_list())
    _turnout_vote_date = {x['voter_id']: x['vote_date'] for x in _turnout_list.to_dict(orient='records')}
    return _turnout_list, _turnout_vuids, _turnout_vote_date


def get_election_results(election_results_file: Path = ELECTION_RESULTS_PATH) -> pd.DataFrame:
    ic("Getting election results.")
    _election_results = pd.read_csv(election_results_file)
    _election_results[WINNER_PERCENT_REF] = round(_election_results[WINNER_PERCENT_REF] * 100, 2)
    return _election_results


ADDR_DICT = dict()
SEEN_ADDRESSES = set()
SEEN_ADDRESS_EDR = dict()
BAD_RECORDS = []
TARGET_COUNT = 0
NON_TARGET_COUNT = 0

texas_vep_start_date = datetime.strptime('10/20/2023', '%m/%d/%Y').date()

previous_years = list(get_previous_vep_year_vuids())
current_year, current_year_vuids = get_current_year_dataframe()

latest_matches_vuids = previous_years + current_year_vuids

all_matched_vuids = pd.DataFrame(
    list(set(latest_matches_vuids)), columns=[VUID_REF.upper(), VEP_YEAR_REF.upper()]).astype(str)

all_matched_vuids[VUID_REF.upper()] = all_matched_vuids[VUID_REF.upper()].astype(int)

vuid_dict = all_matched_vuids.set_index(VUID_REF.upper())[VEP_YEAR_REF.upper()].to_dict()

turnout_list, turnout_vuids, turnout_vote_date = get_turnout_data()

last_month_texas_voterfile = StateVoterFile(state='texas')
last_month = last_month_texas_voterfile.read(VOTERFILE_FOLDER / 'texas/texasnovember2024.csv')
last_month_texas_voterfile.validate(x for x in last_month if int(x['EDR']) >= 20190701)


RECORD_CREATOR = CreateRecords()
validated_records = RECORD_CREATOR.create_records(last_month_texas_voterfile.validation.valid)
record_set = match_records_at_target_address(validated_records)
critical_data = select_only_critical_info(record_set, turnout_dict=turnout_vote_date)

matched_that_voted = (x for x in critical_data if x.vuid in turnout_vuids)
matched_did_not_vote = (x for x in critical_data if x.vuid not in turnout_vuids)


matched_that_voted_df = pd.DataFrame(matched_that_voted)
matched_did_not_vote_df = pd.DataFrame(matched_did_not_vote)

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
).reset_index().fillna(0).replace("", pd.NA).dropna()

excluded_cols = [
    x for x in matched_by_year.columns if x not in [
        COUNTY_REF, SENATE_REF.result_ref.lower(), HOUSE_REF.result_ref.lower(), CONGRESSIONAL.result_ref.lower()
    ]
]

matched_by_year[excluded_cols] = matched_by_year[excluded_cols].astype(int)

election_results = get_election_results()

get_office_results = partial(office_type_results, election_results_df=election_results)
cd_turnout = get_office_results(CONGRESSIONAL)
hd_turnout = get_office_results(HOUSE_REF)
sd_turnout = get_office_results(SENATE_REF)

vep2024_matched_that_voted = matched_that_voted_df[matched_that_voted_df[VEP_YEAR_REF] == 'vep2024']
vep2024_matched_that_voted.to_csv(SHAPE_EXPORTS / 'G2024_vep_voterids.csv', index=False)
