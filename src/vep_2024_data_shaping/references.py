from pathlib import Path
from typing import NamedTuple

PROJECT_FOLDER = Path(__file__).parent.parent.parent.parent
ELECTION_RESULTS_FOLDER_PATH = PROJECT_FOLDER / 'texas-result-scraper' / 'texas_result_scraper' / 'data'
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
G2020_ELECTION_RESULTS_PATH = ELECTION_RESULTS_FOLDER_PATH / "tx-44144-262-statewide-results.csv"
G2020_ELECTION_REF = "TX-2020-GE"
G2022_ELECTION_RESULTS_PATH = ELECTION_RESULTS_FOLDER_PATH / 'tx-47009-242-statewide-results.csv'
G2022_ELECTION_REF = "TX-2022-GE"
G2024_ELECTION_RESULTS_PATH = ELECTION_RESULTS_FOLDER_PATH / 'tx-49664-1010-statewide-results.csv'
G2024_RACE_RESULTS_PATH = ELECTION_RESULTS_FOLDER_PATH / "tx-49664-1010-race-results.csv"


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
    voterfile_ref="congressional district",
    result_ref="CD")

HOUSE_REF = DistrictRef(
    voterfile_ref="legislative lower",
    result_ref="HD")

SENATE_REF = DistrictRef(
    voterfile_ref="legislative upper",
    result_ref="SD")
