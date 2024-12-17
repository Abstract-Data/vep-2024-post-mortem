from __future__ import annotations
from datetime import datetime, date
from typing import Generator, Iterable, ForwardRef
from functools import partial
import itertools
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from pathlib import Path
from icecream import ic
from tqdm import tqdm
import polars as pl

from vep_2024.processing.funcs.loaders.state_loader import StateVoterFile, CreateRecords
from vep_validation_tools.pydantic_models.cleanup_model import PreValidationCleanUp

from vep_2024_data_shaping.references import *
from vep_2024_data_shaping.critical_data import CriticalData

ic.configureOutput(includeContext=False)



""" === FUNCTIONS === """


def create_critical_dict(x: ForwardRef("RecordBaseModel"), _turnout_dict: dict[int, date]) -> dict:
    """
    Creates a dictionary containing critical information from a record.

    Args:
        x (ForwardRef("RecordBaseModel")): A record object containing voter information.
        _turnout_dict (dict[int, date]): A dictionary mapping voter IDs to their voting dates.

    Returns:
        dict: A dictionary containing critical information extracted from the record.
    """
    _x = {}

    vr = x.get('voter_registration')
    if vr:
        _x.update(vr.model_dump(exclude='attributes'))
        _x['dob'] = x.get('name').dob

    _x[VEP_YEAR_REF] = x.get(VEP_YEAR_REF)  # type: ignore

    if _voted := _turnout_dict.get(int(vr.vuid) if vr else None):
        _x['date_voted'] = _voted
        _x['voted_in_2024'] = True

    if _districts := x.get('district_set'):
        for d in _districts.districts:
            _x[d.name] = int(d.number) if d.number else None

    if _vote_history := x.get('vote_history'):
        for election in _vote_history:
            if election.election_id == G2020_ELECTION_REF:
                _x['voted_in_2020'] = True
            elif election.election_id == G2022_ELECTION_REF:
                _x['voted_in_2022'] = True
    return _x


def match_to_address(r: ForwardRef('RecordBaseModel')):
    """
    Matches a record to an address and updates global dictionaries with the matched information.

    Args:
        r (ForwardRef('RecordBaseModel')): A record object containing voter information.

    Returns:
        dict or None: A dictionary containing the matched record information if a match is found, otherwise None.
    """

    global SEEN_ADDRESSES, ADDR_DICT, TARGET_COUNT, NON_TARGET_COUNT
    found, _record = False, None
    if (_vuid := int(r.voter_registration.vuid)) in target_vuid_dict:
        found = True
        _record = dict(r)
        _record[VEP_YEAR_REF] = target_vuid_dict.get(_vuid)
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
    return _record if _record else None


def select_only_critical_info(
        records: Iterable[ForwardRef('RecordBaseModel')],
        turnout_dict: dict[str, date]) -> Generator[CriticalData, None, None]:
    """
     Filters and validates critical information from a list of records.

     Args:
         records (Iterable[ForwardRef('RecordBaseModel')]): An iterable of record objects.
         turnout_dict (dict[str, date]): A dictionary mapping voter IDs to their voting dates.

     Yields:
         CriticalData: Validated critical data objects.
     """
    ic("Filtering critical data.")
    _records = (create_critical_dict(r, turnout_dict) for r in records)
    for _x in _records:
        try:
            _validate = CriticalData(**_x)
            yield _validate
        except Exception as e:
            ic("Bad record found during critical parsing: ", _x)
            BAD_RECORDS.append(e)
            pass
    ic(f"Critical Data filtered with {len(BAD_RECORDS):,} bad records.")


def match_records_at_target_address(
        records: Iterable[PreValidationCleanUp | ForwardRef('RecordBaseModel')]
) -> Generator[dict, None, None]:
    """
    Matches records to target addresses and yields matched records.

    Args:
        records (Iterable[PreValidationCleanUp | ForwardRef('RecordBaseModel')]):
            An iterable of record objects to be matched to target addresses.

    Yields:
        dict: A dictionary containing the matched record information.
    """
    for record in tqdm(records, desc="Matching records to target addresses"):
        if match := match_to_address(record):
            yield match

        if TARGET_COUNT > 0 and TARGET_COUNT % 5000 == 0:
            total = TARGET_COUNT + NON_TARGET_COUNT
            sys.stdout.write(f"""
            Targets: {TARGET_COUNT:,}. 
            Additional: {NON_TARGET_COUNT:,} ({(NON_TARGET_COUNT / total) * 100:.2f}%). 
            Total: {total:,}
    """)
            sys.stdout.flush()


def exclude_cols_from_df(df: pd.DataFrame, cols: list[str] = None) -> list[str]:
    """
    Creates a dictionary containing critical information from a record.

    Args:
        x (ForwardRef("RecordBaseModel")): A record object containing voter information.
        _turnout_dict (dict[int, date]): A dictionary mapping voter IDs to their voting dates.

    Returns:
        dict: A dictionary containing critical information extracted from the record.
    """
    if not cols:
        cols = [
            COUNTY_REF,
            'age_range',
            SENATE_REF.result_ref.lower(),
            HOUSE_REF.result_ref.lower(),
            CONGRESSIONAL.result_ref.lower(),
            VEP_YEAR_REF,
            VUID_REF
        ]
    return [x for x in list(df.columns) if x not in cols]
n


def get_previous_vep_year_vuids(rec_ls: list = None) -> Generator[tuple[int, str], None, None]:
    """
    Retrieves VEP matched voters from previous years.

    Args:
        rec_ls (list, optional): A list of file paths containing VEP matched voters data.
                                 If not provided, a default list is used.

    Yields:
        tuple[int, str]: A tuple containing the VUID and the year of the VEP match.
    """
    ic("Getting previous years VEP matched voters.")
    if not rec_ls:
        rec_ls = VEP_PREVIOUS_YEARS_MATCHED_VOTERS
    for f in rec_ls:
        _data = pd.read_csv(f)[VUID_REF.upper()].map(str).to_list()
        for x in _data:
            yield int(x), f.stem.split('_')[0]


def get_current_year_dataframe(rec_ls: list = None) -> tuple[pd.DataFrame, list[tuple[int, str]]]:
    """
    Retrieves the current year VEP matched voters data and creates a DataFrame and a list of VUIDs.

    Args:
        rec_ls (list, optional): A list of file paths containing VEP matched voters data.
                                 If not provided, a default list is used.

    Returns:
        tuple[pd.DataFrame, list[tuple[int, str]]]: A tuple containing a DataFrame of the current year VEP matched voters
                                                    and a list of tuples with VUIDs and the year 'vep2024'.
    """
    if not rec_ls:
        rec_ls = VEP2024_MATCHED_VOTERS

    ic("Getting current year VEP matched voters.")
    _data = pd.concat([pd.read_csv(f) for f in rec_ls])
    data = _data.drop_duplicates(subset=[VUID_REF])
    vuid_list = [(int(y), 'vep2024') for y in data[VUID_REF].to_list()]
    return data, vuid_list


def get_turnout_data(turnout_file: Path = TURNOUT_LIST_PATH) -> tuple[pd.DataFrame, dict, dict]:
    """
    Retrieves turnout data from a CSV file and creates a DataFrame, a dictionary of voter IDs, and a dictionary of vote dates.

    Args:
        turnout_file (Path, optional): The file path to the CSV file containing turnout data. Defaults to TURNOUT_LIST_PATH.

    Returns:
        tuple[pd.DataFrame, dict, dict]: A tuple containing:
            - A DataFrame of the turnout data.
            - A dictionary with voter IDs as keys and None as values.
            - A dictionary mapping voter IDs to their vote dates.
    """
    ic("Getting turnout data.")
    _turnout_list = pd.read_csv(turnout_file)
    _turnout_vuids = dict.fromkeys(int(x) for x in _turnout_list['voter_id'].to_list())
    _turnout_vote_date = {int(x['voter_id']): x['vote_date'] for x in _turnout_list.to_dict(orient='records')}
    return _turnout_list, _turnout_vuids, _turnout_vote_date



def office_type_results(office: DistrictRef,
                        election_results_df: pd. DataFrame,
                        voted_by_year_df: pd.DataFrame,
                        excluded_cols: list) -> pd.DataFrame:
    ic("Creating turnout data for ", office.result_ref)
    _results = election_results_df[election_results_df[OFFICE_TYPE_REF] == office.result_ref]
    _results[OFFICE_DISTRICT_REF] = _results[OFFICE_DISTRICT_REF].astype(int)
    _matches = voted_by_year_df.groupby(office.result_ref.lower())[excluded_cols].sum().reset_index()
    _turnout = _results.merge(
        _matches,
        right_on=office.result_ref.lower(),
        left_on=OFFICE_DISTRICT_REF
    )
    _turnout = _turnout[_turnout[WINNER_MARGIN_REF] > 0]

    _turnout[VEP_PCT_OF_MARGIN_REF] = np.where(
        _turnout[WINNER_PARTY_REF] == PARTY_FILTER,
        round((_turnout[VEP_TOTAL_VOTERS_REF] / _turnout[WINNER_MARGIN_REF]) * 100, 2),
        np.nan
    )

    _turnout[VEP_WON_RACE_REF] = np.where(
        (_turnout[WINNER_MARGIN_REF] != 0)
        & (_turnout[WINNER_PARTY_REF] == PARTY_FILTER),
        np.select(
            [
                round((_turnout[VEP_TOTAL_VOTERS_REF] / _turnout[WINNER_MARGIN_REF]) * 100, 2) > 100,  # Over 100%
                round((_turnout[VEP_TOTAL_VOTERS_REF] / _turnout[WINNER_MARGIN_REF]) * 100, 2) > 90,  # Over 90%
                round((_turnout[VEP_TOTAL_VOTERS_REF] / _turnout[WINNER_MARGIN_REF]) * 100, 2) > 75,  # Over 75%
                round((_turnout[VEP_TOTAL_VOTERS_REF] / _turnout[WINNER_MARGIN_REF]) * 100, 2) > 50,  # Over 50%
                round((_turnout[VEP_TOTAL_VOTERS_REF] / _turnout[WINNER_MARGIN_REF]) * 100, 2) > 25,  # Over 25%
            ],
            [
                "Over 100%",
                "Over 90%",
                "Over 75%",
                "Over 50%",
                "Over 25%",
            ],
            default=""
        ),
        ""
    )

    _turnout[OFFICE_DISTRICT_REF] = _turnout[OFFICE_DISTRICT_REF].astype(int)
    _cols_to_drop = [x for x in list(_turnout.columns) if x in ['year', 'hd', 'cd', 'sd']]
    _turnout[['vep2020', 'vep2022', 'vep2024']] = _turnout[['vep2020', 'vep2022', 'vep2024']].fillna(0).astype(int)
    _turnout = _turnout.drop(columns=_cols_to_drop)
    _turnout = _turnout.drop_duplicates()
    _transform = _turnout.groupby(['office']).transform('max')['total_votes']
    _turnout = _turnout[_turnout['total_votes'] == _transform]
    return _turnout


def export_office_results(df: pd.DataFrame, office):
    df.sort_values(
        by=OFFICE_DISTRICT_REF
    ).fillna(0).to_csv(
        DOWNLOAD_FOLDER / f'{datetime.today().strftime("%Y%m%d")}_vep2024_{office.result_ref.lower()}_turnout1.csv',
        index=False
    )

def create_matched_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(
        index=[
            df[COUNTY_REF],
            df[SENATE_REF.result_ref.lower()],
            df[HOUSE_REF.result_ref.lower()],
            df[CONGRESSIONAL.result_ref.lower()]
        ],
        columns=df[VEP_YEAR_REF],
        values=df[VUID_REF],
        aggfunc='count',
        margins=True,
        margins_name=VEP_TOTAL_VOTERS_REF
    ).reset_index().fillna(0).replace("", pd.NA).dropna()


# def get_election_results(election_results_file: Path = ELECTION_RESULTS_PATH) -> pd.DataFrame:
#     ic("Getting election results.")
#     _election_results = pd.read_csv(election_results_file)
#     _election_results[WINNER_PERCENT_REF] = round(_election_results[WINNER_PERCENT_REF] * 100, 2)
#     return _election_results


ADDR_DICT = dict()
SEEN_ADDRESSES = set()
SEEN_ADDRESS_EDR = dict()
BAD_RECORDS = []
TARGET_COUNT = 0
NON_TARGET_COUNT = 0
REPORT_DATA_POINTS = dict()

texas_vep_start_date = datetime.strptime('10/20/2023', '%m/%d/%Y').date()

previous_years = list(get_previous_vep_year_vuids())
current_year, current_year_vuids = get_current_year_dataframe()

latest_matches_vuids = previous_years + current_year_vuids

all_matched_vuids = pd.DataFrame(
    list(set(latest_matches_vuids)), columns=[VUID_REF.upper(), VEP_YEAR_REF.upper()]).astype(str)
all_matched_vuids = all_matched_vuids.drop_duplicates(subset=[VUID_REF.upper()])

all_matched_vuids[VUID_REF.upper()] = all_matched_vuids[VUID_REF.upper()].astype(int)

target_vuid_dict = all_matched_vuids.set_index(VUID_REF.upper())[VEP_YEAR_REF.upper()].to_dict()

turnout_list, turnout_vuids, turnout_vote_date = get_turnout_data()


def create_result_merge_table(func):
    def wrapper(*args, **kwargs):
        df = func(*args, **kwargs)
        year = kwargs.get('year')  # Assuming year is the first argument

        # # Columns to be prefixed with year
        # columns_to_prefix = ['total_votes', 'winner', 'winner_party', 'winner_percent', 'winner_margin']
        #
        # # Create new column names with year prefix
        # new_columns = {col: f"{year}_{col}" for col in columns_to_prefix}

        # # Rename the columns in the DataFrame
        # df = df.rename(columns=new_columns)

        # Select only the needed columns, using both original and new column names
        return df[['office',
            'office_type', 'office_district',
            f"total_votes", f"winner",
            f"winner_party", f"winner_percent",
            f"winner_margin", 'year'
        ]]
    return wrapper

@create_result_merge_table
def read_election_results_table(file: Path, **kwargs):
    df = pd.read_csv(file)
    df['office_key'] = df['office_type'].str.lower() + df['office_district'].astype(str).map(str.lower)
    df['year'] = kwargs.get('year')
    return df


g2020_results = read_election_results_table(G2020_ELECTION_RESULTS_PATH, year=2020)
g2022_results = read_election_results_table(G2022_ELECTION_RESULTS_PATH, year=2022)
g2024_results = read_election_results_table(G2024_ELECTION_RESULTS_PATH, year=2024)
# g2024_race_results = pd.read_csv(G2024_RACE_RESULTS_PATH)
# g2024_race_results = g2024_race_results[g2024_race_results['office_type'].isin(['CD', 'HD', 'SD', 'US SENATE', 'POTUS'])]
# g2024_race_results = g2024_race_results.fillna(0)
# g2024_race_results_group = g2024_race_results.groupby(
#     ['office', 'office_type', 'office_district', 'candidate', 'party']
# ).agg({'early_votes': 'sum',
#         'election_day_votes': 'sum',
#         'total_votes': 'sum',
#         'percent_votes': 'mean'}).reset_index()
# g2024_race_results_group = g2024_race_results_group[g2024_race_results_group['percent_votes'] < 100]
all_results = pd.concat([g2020_results, g2022_results, g2024_results], ignore_index=True)

all_results = all_results[all_results['office_type'].isin(['CD', 'SD', 'HD', 'US SENATE', 'POTUS'])]

all_results_ct = all_results.groupby(['office', 'office_type', 'office_district', 'year', 'winner', 'winner_party'], dropna=False).agg({
    'winner_margin': 'sum',
    'winner_percent': 'sum', # Assuming 'winner_margin' is the margin of victory
    'total_votes': 'sum',  # Total votes as context
}).reset_index()

all_results_ct = all_results_ct.sort_values(['office_type', 'office_district', 'year'])

all_results_time = all_results_ct.pivot_table(
    index=['office', 'office_type', 'office_district'],
    columns='year',
    values=['total_votes', 'winner', 'winner_party', 'winner_margin', 'winner_percent'],
    aggfunc={'total_votes': 'sum', 'winner': 'first', 'winner_party': 'first', 'winner_margin': 'first', 'winner_percent': 'first'}
).reset_index()

winner_margin_cols = [x for x in list(all_results_time.columns) if 'winner_margin' in x]
winner_percent_cols = [x for x in list(all_results_time.columns) if 'winner_percent' in x]
total_votes_cols = [x for x in list(all_results_time.columns) if 'total_votes' in x]
all_results_time[winner_margin_cols] = all_results_time[winner_margin_cols].fillna(0).astype(int)
all_results_time[winner_percent_cols] = all_results_time[winner_percent_cols].fillna(0.0).astype(float)
all_results_time[total_votes_cols] = all_results_time[total_votes_cols].fillna(0).astype(int)

all_results_time_flatten = all_results_time.copy()
# Convert all parts of each MultiIndex level to strings
all_results_time_flatten.columns = pd.MultiIndex.from_tuples(
    [tuple(str(item) for item in col) for col in all_results_time_flatten.columns]
)

# Now flatten the MultiIndex columns
all_results_time_flatten.columns = ['_'.join(col).strip() if len(col) > 1 else col[0] for col in all_results_time_flatten.columns.values]
all_results_time_rename_cols = {col: col[:-1] for col in all_results_time_flatten.columns if col.endswith('_')}
all_results_time_flatten = all_results_time_flatten.rename(columns=all_results_time_rename_cols)
all_results_time_flatten = all_results_time_flatten.fillna(pd.NA)
all_results_group = all_results_time_flatten.groupby(
    ['office', 'office_type', 'office_district']
).agg({
    'total_votes_2020': 'sum',
    'total_votes_2022': 'sum',
    'total_votes_2024': 'sum',
    'winner_2020': 'first',
    'winner_2022': 'first',
    'winner_2024': 'first',
    'winner_party_2020': 'first',
    'winner_party_2022': 'first',
    'winner_party_2024': 'first',
    'winner_margin_2020': 'sum',
    'winner_margin_2022': 'sum',
    'winner_margin_2024': 'sum',
    'winner_percent_2020': 'sum',
    'winner_percent_2022': 'sum',
    'winner_percent_2024': 'sum'
}).reset_index()

all_results_2020 = all_results_group[all_results_group['winner_percent_2020'] < 100]
all_results_2022 = all_results_group[all_results_group['winner_percent_2022'] < 100]
all_results_2024 = all_results_group[all_results_group['winner_percent_2024'] < 100]

filtered_results = all_results_group[
    ((all_results_group['winner_party_2020'] == 'R') &
     (all_results_group['winner_margin_2020'] > 0)) |
    ((all_results_group['winner_party_2022'] == 'R') &
     (all_results_group['winner_margin_2022'] > 0)) |
    ((all_results_group['winner_party_2024'] == 'R') &
     (all_results_group['winner_margin_2024'] > 0))
    ]
# last_month_texas_voterfile = StateVoterFile(state='texas')
# last_month = last_month_texas_voterfile.read(VOTERFILE_FOLDER / 'texas/texasnovember2024.csv')
# last_month_texas_voterfile.validate(x for x in last_month if int(x['EDR']) >= 20191101)

# new_voters_since_2019 = pl.scan_csv(VOTERFILE_FOLDER / 'texas/texasnovember2024.csv', infer_schema=False)
# new_voters_since_2019 = new_voters_since_2019.with_columns(
#     pl.col('EDR').str.strptime(pl.Date, '%Y%m%d')
# )
# new_voters_by_year = (
#     new_voters_since_2019
#     .filter(pl.col('EDR') >= pl.Date('2019-11-01'))
#     .with_column(pl.col('EDR').dt.year().alias('year'))
#     .groupby('year')
#     .count()
# )
#
# # test = next(last_month_texas_voterfile.validation.valid)
# RECORD_CREATOR = CreateRecords()
# validated_records = RECORD_CREATOR.create_records(last_month_texas_voterfile.validation.valid)
# record_set = match_records_at_target_address(validated_records)
# critical_data = select_only_critical_info(record_set, turnout_dict=turnout_vote_date)
#
# matched_that_voted = (x for x in critical_data if x.voted_in_2024)
# matched_did_not_vote = (x for x in critical_data if not x.voted_in_2024)
#
# matched_that_voted_df = pd.DataFrame(matched_that_voted)
# matched_that_voted_df.to_csv(SHAPE_EXPORTS / '2024_texas_matched_voted.csv', index=False)

#
# targets_in_2024 = pd.concat([matched_that_voted_df, matched_did_not_vote_df])
# targets_in_2024 = targets_in_2024[targets_in_2024[VEP_YEAR_REF] == 'vep2024'].drop_duplicates(subset=[VUID_REF])
# matched_that_voted_df.to_csv(SHAPE_EXPORTS / '2024_texas_matched_voted.csv', index=False)
#
matched_that_voted_df = pd.read_csv(SHAPE_EXPORTS / '2024_texas_matched_voted.csv')
exclude_cols = ['voted_in_2020', 'voted_in_2022', 'voted_in_2024', 'precinct_name', 'precinct_number', 'date_voted', 'age', 'edr', 'id', 'dob']
matched_that_voted_df = matched_that_voted_df.drop(columns=exclude_cols)

matched_crosstab = create_matched_crosstab(matched_that_voted_df)

district_excluded_cols = exclude_cols_from_df(matched_crosstab)

office_results = partial(
    office_type_results,
    voted_by_year_df=matched_crosstab,
    election_results_df=g2024_results,
    excluded_cols=district_excluded_cols
)
#
# g2020_turnout = get_office_results(office=CONGRESSIONAL, election_results_df=g2020_results)
cd_turnout = office_results(CONGRESSIONAL)
hd_turnout = office_results(HOUSE_REF)
sd_turnout = office_results(SENATE_REF)

today_date = datetime.today().strftime('%Y%m%d')
cd_turnout.to_csv(SHAPE_EXPORTS / f'{today_date}_cd_turnout.csv', index=False)
hd_turnout.to_csv(SHAPE_EXPORTS / f'{today_date}_hd_turnout.csv', index=False)
sd_turnout.to_csv(SHAPE_EXPORTS / f'{today_date}_sd_turnout.csv', index=False)

cd_vep_won = cd_turnout[cd_turnout[VEP_WON_RACE_REF] != ""]
hd_vep_won = hd_turnout[hd_turnout[VEP_WON_RACE_REF] != ""]
sd_vep_won = sd_turnout[sd_turnout[VEP_WON_RACE_REF] != ""]

new_registrations_since_august_2024 = all_matched_vuids.count() - 535_986
vep2022_voters_who_voted = round(matched_that_voted_df[matched_that_voted_df['vep_year'] == 'vep2022']['vuid'].count() / 169_160, 4)
vep2020_voters_who_voted = round(matched_that_voted_df[matched_that_voted_df['vep_year'] == 'vep2020']['vuid'].count() / 212_972, 4)
vep2024_voters_who_voted = round(matched_that_voted_df[matched_that_voted_df['vep_year'] == 'vep2024']['vuid'].count() / 197_285, 4)
total_vep_votes = matched_that_voted_df['vuid'].count()
g2024_victory_margin = 6_393_597 - 5_113_725
g2024_compared_to_2018 = g2024_victory_margin - 602_120
trump2016_votes = 4_685_047
trump2020_votes = 5_890_347
trump2024_votes = 6_393_597
trump2024_margin = trump2024_votes - trump2020_votes
trump2024_margin_compared_to_2016 = trump2024_votes - trump2016_votes
vep_margin_trump_difference_compared_to_2024 =  total_vep_votes / trump2024_margin
vep_margin_trump_difference_compared_to_2020 = total_vep_votes / trump2024_margin_compared_to_2016

ted_cruz_2018 = 4_260_553
ted_cruz_2024 = 5_990_741
ted_cruz_margin = ted_cruz_2024 - ted_cruz_2018
vep_margin_cruz_difference_compared_to_2024 = total_vep_votes / ted_cruz_margin

print("Trump 2024 Votes: ", trump2024_votes)
print("Trump 2024 Margin: ", trump2024_margin)
print("Trump 2024 Margin Compared to 2016: ", trump2024_margin_compared_to_2016)
print("VEP Margin Trump Difference Compared to 2024: ", vep_margin_trump_difference_compared_to_2024)
print("VEP Margin Trump Difference Compared to 2020: ", vep_margin_trump_difference_compared_to_2020)
print("Ted Cruz 2024 Votes: ", ted_cruz_2024)
print("Ted Cruz Margin: ", ted_cruz_margin)
print("VEP Margin Cruz Difference Compared to 2024: ", vep_margin_cruz_difference_compared_to_2024)
print("New Registrations Since August 2024: ", new_registrations_since_august_2024)
print("VEP2022 Voters Who Voted: ", vep2022_voters_who_voted)
print("VEP2020 Voters Who Voted: ", vep2020_voters_who_voted)
print("VEP2024 Voters Who Voted: ", vep2024_voters_who_voted)
print("Total VEP Votes: ", total_vep_votes)
print("G2024 Victory Margin: ", g2024_victory_margin)
print("G2024 Compared to 2018: ", g2024_compared_to_2018)

#
# vep2024_matched_that_voted = matched_that_voted_df[matched_that_voted_df[VEP_YEAR_REF] == 'vep2024']
# vep2024_matched_that_voted.to_csv(SHAPE_EXPORTS / 'G2024_vep_voterids.csv', index=False)
