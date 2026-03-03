"""
Generate persona JSON files from ANES 2024 Time Series data.

Selects respondents with complete data on key demographics, then builds
10-module persona files matching the framework's modular database schema.
Known attitudes from selected ANES items are stored in the relevant modules;
evaluation items are strictly held out.

Usage:
    python scripts/generate_personas.py
    python scripts/generate_personas.py --n 10
    python scripts/generate_personas.py --n 50 --seed 42
"""

import csv
import json
import os
import sys
import argparse
import random
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = DEMO_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "anes"
PERSONAS_DIR = DEMO_DIR / "personas"

ANES_CSV = DATA_DIR / "anes_timeseries_2024_csv_20250808.csv"

# ---------------------------------------------------------------------------
# Sentinel for feeling-thermometer variables (0-100 scale)
# ---------------------------------------------------------------------------

THERMO = "THERMOMETER"


def thermo_label(val):
    n = int(val)
    if n >= 85:
        return f"Very warm ({n}/100)"
    if n >= 60:
        return f"Warm ({n}/100)"
    if n == 50:
        return f"Neutral ({n}/100)"
    if n >= 40:
        return f"Cool ({n}/100)"
    if n >= 15:
        return f"Cold ({n}/100)"
    return f"Very cold ({n}/100)"


# ---------------------------------------------------------------------------
# ANES 2024 variable mappings (confirmed from codebook)
# ---------------------------------------------------------------------------

GENDER_LABELS = {"1": "Male", "2": "Female"}

RACE_LABELS = {
    "1": "White, non-Hispanic", "2": "Black, non-Hispanic",
    "3": "Hispanic", "4": "Asian or Pacific Islander, non-Hispanic",
    "5": "Native American or other race, non-Hispanic",
    "6": "Multiple races, non-Hispanic",
}

RELIGION_LABELS = {
    "1": "Protestant", "2": "Roman Catholic",
    "3": "Orthodox Christian", "4": "Latter-Day Saints",
    "5": "Jewish", "6": "Muslim", "7": "Buddhist", "8": "Hindu",
    "9": "Atheist", "10": "Agnostic", "11": "Something else",
    "12": "Nothing in particular",
}

INCOME_6_LABELS = {
    "1": "Under $10,000", "2": "$10,000-$29,999",
    "3": "$30,000-$59,999", "4": "$60,000-$99,999",
    "5": "$100,000-$249,999", "6": "$250,000 or more",
}

PARTY_ID_LABELS = {
    "1": "Strong Democrat", "2": "Not very strong Democrat",
    "3": "Independent, leans Democratic", "4": "Independent",
    "5": "Independent, leans Republican",
    "6": "Not very strong Republican", "7": "Strong Republican",
}

REGION_LABELS = {"1": "Northeast", "2": "Midwest", "3": "South", "4": "West"}

APPROVE_PRES_LABELS = {
    "1": "Approve strongly", "2": "Approve not strongly",
    "3": "Disapprove not strongly", "4": "Disapprove strongly",
}

IDEOLOGY_LABELS = {
    "1": "Extremely liberal", "2": "Liberal", "3": "Slightly liberal",
    "4": "Moderate", "5": "Slightly conservative",
    "6": "Conservative", "7": "Extremely conservative",
}

HEALTH_INS_LABELS = {"1": "Yes", "2": "No"}

HEALTH_CONCERN_LABELS = {
    "1": "Not at all concerned", "2": "A little concerned",
    "3": "Moderately concerned", "4": "Very concerned",
    "5": "Extremely concerned",
}

FINANCIAL_WORRY_LABELS = {
    "1": "Extremely worried", "2": "Very worried",
    "3": "Moderately worried", "4": "A little worried",
    "5": "Not at all worried",
}

MARITAL_LABELS = {
    "1": "Married", "2": "Widowed", "3": "Divorced",
    "4": "Separated", "5": "Never married",
}

EDUCATION_LABELS = {
    "1": "High school or less", "2": "Some college",
    "3": "Associate degree", "4": "Bachelor's degree or higher",
    "5": "Post-graduate degree",
}

POLITICAL_ATTENTION_LABELS = {
    "1": "Always", "2": "Most of the time", "3": "About half the time",
    "4": "Some of the time", "5": "Never",
}

CAMPAIGN_INTEREST_LABELS = {
    "1": "Very much interested", "2": "Somewhat interested",
    "3": "Not much interested",
}

REGISTERED_LABELS = {"1": "Yes", "2": "No"}

GOVT_HEALTH_LABELS = {
    "1": "Government should ensure coverage",
    "2": "Not the government's responsibility",
}

DEPRESSION_LABELS = {
    "1": "None of the time", "2": "Some of the time",
    "3": "About half the time", "4": "Most of the time",
    "5": "All of the time",
}

PC_SENSITIVITY_LABELS = {
    "1": "The way people talk needs to change a lot",
    "2": "The way people talk needs to change a little",
    "3": "People are a little too easily offended",
    "4": "People are much too easily offended",
}

SELF_CENSOR_LABELS = {
    "1": "Never", "2": "Rarely", "3": "Occasionally",
    "4": "Fairly often", "5": "Very often",
}

VIOLENCE_JUSTIFIED_LABELS = {
    "1": "Not at all", "2": "A little", "3": "A moderate amount",
    "4": "A lot", "5": "A great deal",
}

VIOLENCE_CHANGE_LABELS = {
    "1": "Increased", "2": "Decreased", "3": "Stayed the same",
}

TRUST_GOVT_LABELS = {
    "1": "Always", "2": "Most of the time",
    "3": "About half the time", "4": "Some of the time", "5": "Never",
}

IMMIGRATION_7PT_LABELS = {
    "1": "Increase a great deal", "2": "Increase a moderate amount",
    "3": "Increase a little", "4": "Keep about the same",
    "5": "Decrease a little", "6": "Decrease a moderate amount",
    "7": "Decrease a great deal",
}

BIG_INTERESTS_LABELS = {
    "1": "Run by a few big interests",
    "2": "For the benefit of all the people",
}

GOVT_WASTE_LABELS = {
    "1": "Waste a lot", "2": "Waste some", "3": "Don't waste very much",
}

# ---------------------------------------------------------------------------
# NEW label maps for enriched modules
# ---------------------------------------------------------------------------

YES_NO_LABELS = {"1": "Yes", "2": "No"}

BINARY_01_LABELS = {"0": "No", "1": "Yes"}

DEFENSE_SPENDING_7PT_LABELS = {
    "1": "Greatly decrease defense spending",
    "2": "Decrease defense spending",
    "3": "Slightly decrease defense spending",
    "4": "Keep defense spending about the same",
    "5": "Slightly increase defense spending",
    "6": "Increase defense spending",
    "7": "Greatly increase defense spending",
}

GUARANTEED_JOBS_7PT_LABELS = {
    "1": "Government should ensure a job and good standard of living",
    "2": "2", "3": "3", "4": "4", "5": "5", "6": "6",
    "7": "Each person should get ahead on their own",
}

ENVIRO_JOBS_7PT_LABELS = {
    "1": "Protect environment even if it costs jobs and standard of living",
    "2": "2", "3": "3", "4": "4", "5": "5", "6": "6",
    "7": "Prioritize jobs even if environment suffers",
}

EMPLOYMENT_LABELS = {
    "1": "Working now", "2": "Temporarily laid off",
    "3": "Unemployed", "4": "Retired",
    "5": "Permanently disabled", "6": "Homemaker",
    "7": "Student", "8": "Other",
}

WORK_SCHEDULE_LABELS = {
    "1": "Full time", "2": "Part time", "3": "Varies",
}

WORK_SECTOR_LABELS = {
    "1": "Government", "2": "Private sector", "3": "Non-profit",
    "4": "Self-employed",
}

HOMEOWNER_LABELS = {
    "1": "Own or buying", "2": "Rent", "3": "Other arrangement",
}

UNION_LABELS = {
    "1": "Yes, current member", "2": "No", "3": "Former member",
}

FOOD_SECURITY_LABELS = {
    "1": "Enough of the kinds of food we want",
    "2": "Enough but not always the kinds we want",
    "3": "Sometimes not enough to eat",
    "4": "Often not enough to eat",
}

BILL_DIFFICULTY_LABELS = {
    "1": "Not difficult at all", "2": "Not very difficult",
    "3": "Somewhat difficult", "4": "Very difficult",
    "5": "Extremely difficult",
}

ECONOMY_FUTURE_LABELS = {
    "1": "Better", "2": "About the same", "5": "Worse",
}

INFLATION_LABELS = {
    "1": "Increased a lot", "2": "Increased somewhat",
    "3": "Stayed about the same",
    "4": "Decreased somewhat", "5": "Decreased a lot",
}

PERSONAL_FIN_FUTURE_LABELS = {
    "1": "Better off", "2": "About the same", "5": "Worse off",
}

SOCIAL_TRUST_LABELS = {
    "1": "Most people can be trusted",
    "2": "You can't be too careful in dealing with people",
}

FAVOR_OPPOSE_5PT_LABELS = {
    "1": "Favor a great deal", "2": "Favor somewhat",
    "3": "Neither favor nor oppose",
    "4": "Oppose somewhat", "5": "Oppose a great deal",
}

FAVOR_OPPOSE_LABELS = {"1": "Favor", "2": "Oppose"}

AUTH_INDEPENDENCE_LABELS = {
    "0": "Independence", "50": "Both equally", "100": "Respect for elders",
}

AUTH_CURIOSITY_LABELS = {
    "0": "Curiosity", "50": "Both equally", "100": "Good manners",
}

AUTH_SELFRELIANCE_LABELS = {
    "0": "Self-reliance", "50": "Both equally", "100": "Obedience",
}

AUTH_CONSIDERATE_LABELS = {
    "0": "Being considerate", "50": "Both equally", "100": "Being well-behaved",
}

SCIENCE_GOOD_HARM_LABELS = {
    "1": "Benefits of science outweigh the harms",
    "2": "Benefits and harms of science are about equal",
    "3": "Harms of science outweigh the benefits",
}

SCIENCE_REGULATE_LABELS = {
    "1": "Government should regulate business use of science and technology",
    "2": "Government should not regulate business use of science and technology",
}

SCIENCE_TRUST_LABELS = {
    "1": "A great deal of confidence",
    "2": "Only some confidence",
    "3": "Hardly any confidence",
}

NEWS_DAYS_LABELS = {
    "0": "0 days", "1": "1 day", "2": "2 days", "3": "3 days",
    "4": "4 days", "5": "5 days", "6": "6 days", "7": "7 days",
}

SOCIAL_MEDIA_FREQ_LABELS = {
    "1": "Several times a day", "2": "About once a day",
    "3": "A few times a week", "4": "About once a week",
    "5": "Less than once a week", "6": "Never",
}

POLITICAL_NEWS_INTEREST_LABELS = {
    "1": "Very interested", "2": "Somewhat interested",
    "3": "Not very interested", "4": "Not at all interested",
}

DISCRIM_AMOUNT_LABELS = {
    "1": "A great deal", "2": "A lot", "3": "A moderate amount",
    "4": "A little", "5": "None at all",
}

POLICE_TREAT_LABELS = {
    "1": "Treat whites much better than blacks",
    "2": "Treat whites somewhat better",
    "3": "Treat both the same",
    "4": "Treat blacks somewhat better",
    "5": "Treat blacks much better than whites",
}

RACE_INFLUENCE_HIRING_LABELS = {
    "1": "A great deal", "2": "A lot", "3": "A moderate amount",
    "4": "A little", "5": "None at all",
}

CHINA_TARIFFS_LABELS = {
    "1": "Favor a great deal", "2": "Favor somewhat",
    "3": "Neither favor nor oppose",
    "4": "Oppose somewhat", "5": "Oppose a great deal",
}

TRADE_POLICY_LABELS = {
    "1": "Favor increasing trade with other countries",
    "2": "Favor limiting trade with other countries",
}

KNOWLEDGE_LABELS = {
    "1": "Very high (4/4 correct)", "2": "High (3/4 correct)",
    "3": "Medium (2/4 correct)", "4": "Low (0-1/4 correct)",
}

CARE_WHO_WINS_LABELS = {"1": "Care a good deal", "2": "Don't care very much"}

VOTED_LABELS = {"1": "Voted", "2": "Did not vote"}

VOTED_FOR_LABELS = {"1": "Harris", "2": "Trump", "5": "Other candidate"}

PARTY_MEMBER_LABELS = {"1": "Yes", "2": "No"}

BETTER_ECON_PARTY_LABELS = {"1": "Democrats", "2": "Republicans"}

TRAIT_5PT_LABELS = {
    "1": "Extremely well", "2": "Very well", "3": "Moderately well",
    "4": "Slightly", "5": "Not well at all",
}

FAVORABILITY_4PT_LABELS = {
    "1": "Very favorable", "2": "Somewhat favorable",
    "3": "Somewhat unfavorable", "4": "Very unfavorable",
}

ELECTION_FAIR_LABELS = {"1": "Votes counted fairly", "2": "Votes not counted fairly"}

ELECTION_CONFIDENT_LABELS = {"1": "Confident", "2": "Not confident"}

DEMOCRACY_SAT_LABELS = {
    "1": "Very satisfied", "2": "Fairly satisfied",
    "3": "Not very satisfied", "4": "Not at all satisfied", "5": "Not at all satisfied",
}

SYSTEM_FAIR_LABELS = {
    "1": "Treat all people fairly", "2": "Favor the powerful",
    "3": "Neither", "4": "Favor the powerful", "5": "Strongly favor the powerful",
}

NATIONAL_ECON_NOW_LABELS = {
    "1": "Very good", "2": "Good", "3": "Neither good nor bad", "4": "Bad",
}

SELF_HEALTH_LABELS = {
    "1": "Poor", "2": "Fair", "3": "Good", "4": "Very good", "5": "Excellent",
}

LIFE_SAT_LABELS = {
    "1": "Very satisfied", "2": "Fairly satisfied",
    "3": "Not very satisfied", "4": "Not at all satisfied",
}

HARRIS_KNOWLEDGEABLE_LABELS = {"1": "Yes", "2": "No"}
HARRIS_STRONG_LEADER_LABELS = {"1": "Yes", "2": "No"}

SOCIAL_MEDIA_HOURS_LABELS = {
    "1": "None", "2": "Less than 30 minutes", "3": "30 min to 1 hour",
    "4": "1 to 2 hours", "5": "2 to 3 hours", "6": "3 to 4 hours",
    "7": "More than 4 hours",
}

FOX_CNN_FREQ_LABELS = {
    "1": "A great deal", "2": "Quite a bit",
    "3": "Some", "4": "Very little", "5": "None",
}

POLICE_FUNDING_LABELS = {
    "1": "Increase a great deal", "2": "Increase somewhat",
    "3": "Keep the same", "4": "Decrease somewhat",
}

AFFIRM_ACTION_LABELS = {
    "1": "Favor strongly", "2": "Favor not strongly",
    "3": "Oppose not strongly",
}

RACIAL_EQUALITY_LABELS = {
    "1": "Too much attention to racial issues",
    "2": "About the right amount",
    "3": "Not enough attention to racial issues",
}

RACIAL_RESENTMENT_DESERVE_LABELS = {
    "1": "Agree strongly", "2": "Agree somewhat",
    "3": "Neither agree nor disagree",
    "4": "Disagree somewhat", "5": "Disagree strongly",
}

SPENDING_PRIORITY_LABELS = {
    "1": "Increased a lot", "2": "Increased somewhat",
    "3": "Kept about the same",
    "4": "Decreased somewhat", "5": "Decreased a lot",
    "6": "Cut out entirely", "7": "Cut out entirely",
}

GUN_PERMIT_LABELS = {
    "1": "Favor a great deal", "2": "Favor somewhat",
    "3": "Oppose somewhat", "4": "Oppose a great deal",
}

TRANSGENDER_POLICY_LABELS = {
    "1": "Favor a great deal", "2": "Favor somewhat",
    "3": "Neither favor nor oppose",
    "4": "Oppose somewhat", "5": "Oppose a great deal",
    "6": "Not sure", "7": "Not sure",
}

DRUG_PRICING_LABELS = {"1": "Favor", "2": "Oppose", "5": "Neither"}

TRADE_GOOD_BAD_LABELS = {"1": "Good", "2": "Bad"}

CHILDREN_BINARY_LABELS = {"1": "Yes, children under 18", "2": "No children under 18"}

MEDICARE_EXPAND_LABELS = {
    "1": "Favor a great deal", "2": "Favor somewhat",
    "3": "Neither favor nor oppose",
    "4": "Oppose somewhat", "5": "Oppose a great deal",
}

RELIGION_ATTENDANCE_LABELS = {
    "1": "Every week", "2": "Almost every week",
    "3": "Once or twice a month", "4": "A few times a year",
    "5": "Seldom", "6": "Never",
}

RELIGION_IMPORTANCE_LABELS = {
    "1": "Very important", "2": "Somewhat important",
    "3": "Not very important", "4": "Not important at all",
}

RELIGION_GUIDANCE_LABELS = {
    "1": "A great deal", "2": "Quite a bit",
    "3": "Some", "4": "None",
}

CHILDREN_IN_HOME_LABELS = {
    "0": "None", "1": "1 child", "2": "2 children",
    "3": "3 children", "4": "4 or more children",
}


FIPS_STATE = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
    "06": "California", "08": "Colorado", "09": "Connecticut",
    "10": "Delaware", "11": "Washington DC", "12": "Florida",
    "13": "Georgia", "15": "Hawaii", "16": "Idaho", "17": "Illinois",
    "18": "Indiana", "19": "Iowa", "20": "Kansas", "21": "Kentucky",
    "22": "Louisiana", "23": "Maine", "24": "Maryland",
    "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
    "28": "Mississippi", "29": "Missouri", "30": "Montana",
    "31": "Nebraska", "32": "Nevada", "33": "New Hampshire",
    "34": "New Jersey", "35": "New Mexico", "36": "New York",
    "37": "North Carolina", "38": "North Dakota", "39": "Ohio",
    "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania",
    "44": "Rhode Island", "45": "South Carolina", "46": "South Dakota",
    "47": "Tennessee", "48": "Texas", "49": "Utah", "50": "Vermont",
    "51": "Virginia", "53": "Washington", "54": "West Virginia",
    "55": "Wisconsin", "56": "Wyoming",
}

# Variables required for selection (must be non-missing)
REQUIRED_VARS = [
    "V241458x", "V241550", "V241501x", "V241461x",
    "V243007", "V241227x", "V241422", "V241567x", "V241465x",
]

# ---------------------------------------------------------------------------
# Module attitude items (stored in persona, NOT used for evaluation)
# Each entry: variable -> (field_name, label_map_or_THERMO)
# ---------------------------------------------------------------------------

MODULE_ITEMS = {
    "politics": {
        # Political identity and evaluations
        "V241004":  ("political_attention", POLITICAL_ATTENTION_LABELS),
        "V241005":  ("campaign_interest", CAMPAIGN_INTEREST_LABELS),
        "V241012":  ("registered_to_vote", REGISTERED_LABELS),
        "V241100":  ("president_approval", APPROVE_PRES_LABELS),
        "V241227x": ("party_identification", PARTY_ID_LABELS),
        "V241102x": ("ideology", IDEOLOGY_LABELS),
        "V241008x": ("political_knowledge_score", KNOWLEDGE_LABELS),
        "V241009":  ("cares_who_wins_election", CARE_WHO_WINS_LABELS),
        "V241103":  ("member_of_political_party", PARTY_MEMBER_LABELS),
        # Voting behavior
        "V241052":  ("voted_2024", VOTED_LABELS),
        "V241053":  ("voted_for_candidate", VOTED_FOR_LABELS),
        # Candidate evaluations
        "V241121":  ("trump_trait_honest", TRAIT_5PT_LABELS),
        "V241122":  ("trump_trait_cares_about_people", TRAIT_5PT_LABELS),
        "V241123":  ("trump_trait_knowledgeable", TRAIT_5PT_LABELS),
        "V241124":  ("trump_trait_strong_leader", TRAIT_5PT_LABELS),
        "V241125":  ("harris_trait_honest", TRAIT_5PT_LABELS),
        "V241126":  ("harris_trait_cares_about_people", TRAIT_5PT_LABELS),
        "V241127":  ("harris_trait_knowledgeable", HARRIS_KNOWLEDGEABLE_LABELS),
        "V241128":  ("harris_trait_strong_leader", HARRIS_STRONG_LEADER_LABELS),
        "V241129x": ("trump_overall_favorability", FAVORABILITY_4PT_LABELS),
        "V241133x": ("harris_overall_favorability", FAVORABILITY_4PT_LABELS),
        # Election legitimacy
        "V241130":  ("election_votes_counted_fairly", ELECTION_FAIR_LABELS),
        "V241131":  ("election_confident_in_count", ELECTION_CONFIDENT_LABELS),
        # Democratic satisfaction
        "V241228":  ("satisfaction_with_democracy", DEMOCRACY_SAT_LABELS),
        "V241230":  ("system_treats_people_fairly", SYSTEM_FAIR_LABELS),
        # Feeling thermometers for political entities
        "V241001":  ("feeling_biden", THERMO),
        "V241002":  ("feeling_trump", THERMO),
        "V241003":  ("feeling_harris", THERMO),
        "V241006":  ("feeling_democratic_party", THERMO),
        "V241007":  ("feeling_republican_party", THERMO),
        "V241013":  ("feeling_congress", THERMO),
        "V241014":  ("feeling_federal_government", THERMO),
        "V241020":  ("feeling_supreme_court", THERMO),
        # Issue positions on govt role
        "V241118":  ("defense_spending_position", DEFENSE_SPENDING_7PT_LABELS),
        "V241119":  ("guaranteed_jobs_position", GUARANTEED_JOBS_7PT_LABELS),
        # Political participation
        "V241521":  ("tried_to_persuade_others_vote", YES_NO_LABELS),
        "V241524":  ("attended_political_rally", YES_NO_LABELS),
        "V241526":  ("donated_to_party_or_candidate", YES_NO_LABELS),
        "V241527":  ("contacted_government_official", YES_NO_LABELS),
        "V241528":  ("participated_in_protest", YES_NO_LABELS),
        "V241531":  ("posted_political_content_online", YES_NO_LABELS),
    },
    "economy": {
        "V241567x": ("income_bracket", INCOME_6_LABELS),
        "V241539":  ("financial_worry", FINANCIAL_WORRY_LABELS),
        # Employment details
        "V241532":  ("employment_status", EMPLOYMENT_LABELS),
        "V241534":  ("work_schedule", WORK_SCHEDULE_LABELS),
        "V241533":  ("work_sector", WORK_SECTOR_LABELS),
        # Assets and financial situation
        "V241536":  ("housing_status", HOMEOWNER_LABELS),
        "V241537":  ("has_stock_investments", YES_NO_LABELS),
        "V241538":  ("union_membership", UNION_LABELS),
        "V241540":  ("food_security", FOOD_SECURITY_LABELS),
        "V241541":  ("difficulty_paying_bills", BILL_DIFFICULTY_LABELS),
        # Economic outlook
        "V241106x": ("national_economy_current", NATIONAL_ECON_NOW_LABELS),
        "V241109":  ("national_economy_next_year", ECONOMY_FUTURE_LABELS),
        "V241111":  ("inflation_assessment", INFLATION_LABELS),
        "V241114":  ("personal_finances_next_year", PERSONAL_FIN_FUTURE_LABELS),
        "V241107":  ("better_economy_under_party", BETTER_ECON_PARTY_LABELS),
        "V241115":  ("govt_services_spending_self_placement", {"1": "Fewer services, reduce spending", "2": "More services, increase spending"}),
        # Feelings about economic groups
        "V241021":  ("feeling_big_business", THERMO),
        "V241022":  ("feeling_labor_unions", THERMO),
        "V241023":  ("feeling_rich_people", THERMO),
        "V241024":  ("feeling_poor_people", THERMO),
        # Trade views
        "V241185":  ("china_tariffs_view", CHINA_TARIFFS_LABELS),
        "V241186":  ("trade_policy_preference", TRADE_POLICY_LABELS),
        "V241213":  ("trade_agreements_good_bad", TRADE_GOOD_BAD_LABELS),
    },
    "health": {
        "V241214":  ("govt_health_insurance_view", GOVT_HEALTH_LABELS),
        "V241571":  ("has_health_insurance", HEALTH_INS_LABELS),
        "V241572":  ("concern_losing_insurance", HEALTH_CONCERN_LABELS),
        "V241573":  ("concern_paying_healthcare", HEALTH_CONCERN_LABELS),
        "V241575":  ("recent_low_interest_pleasure", DEPRESSION_LABELS),
        "V241576":  ("recent_feeling_depressed", DEPRESSION_LABELS),
        "V241570":  ("self_reported_health", SELF_HEALTH_LABELS),
        "V241581":  ("life_satisfaction", LIFE_SAT_LABELS),
        "V241209":  ("medicare_expansion_view", MEDICARE_EXPAND_LABELS),
        "V241211":  ("drug_pricing_policy", DRUG_PRICING_LABELS),
        # Health conditions (ever diagnosed)
        "V241574a": ("condition_asthma", BINARY_01_LABELS),
        "V241574b": ("condition_diabetes", BINARY_01_LABELS),
        "V241574c": ("condition_heart_disease", BINARY_01_LABELS),
        "V241574d": ("condition_high_blood_pressure", BINARY_01_LABELS),
        "V241574e": ("condition_stroke", BINARY_01_LABELS),
        "V241574f": ("condition_cancer", BINARY_01_LABELS),
        "V241574g": ("condition_arthritis", BINARY_01_LABELS),
        "V241574h": ("condition_chronic_pain", BINARY_01_LABELS),
        "V241574i": ("condition_anxiety_disorder", BINARY_01_LABELS),
        "V241574j": ("condition_depression_diagnosed", BINARY_01_LABELS),
    },
    "social_context": {
        "V241577":  ("pc_sensitivity_view", PC_SENSITIVITY_LABELS),
        "V241578":  ("self_censoring_frequency", SELF_CENSOR_LABELS),
        "V241579":  ("political_violence_justified", VIOLENCE_JUSTIFIED_LABELS),
        "V241580":  ("political_violence_change", VIOLENCE_CHANGE_LABELS),
        # Social trust
        "V241156":  ("trust_in_people", SOCIAL_TRUST_LABELS),
        "V241157":  ("people_helpful_vs_selfish", THERMO),
        "V241158":  ("people_fair_vs_take_advantage", THERMO),
        # Feelings toward social groups
        "V241018":  ("feeling_feminists", THERMO),
        "V241019":  ("feeling_people_on_welfare", THERMO),
        "V241042":  ("feeling_lgbtq", THERMO),
        "V241034":  ("feeling_transgender_people", THERMO),
        "V241030":  ("feeling_muslims", THERMO),
        "V241031":  ("feeling_christians", THERMO),
        "V241032":  ("feeling_jews", THERMO),
        "V241033":  ("feeling_atheists", THERMO),
        "V241041":  ("feeling_gun_owners", THERMO),
        "V241044":  ("feeling_united_states", THERMO),
        # Immigration policy positions
        "V241179":  ("border_wall_position", FAVOR_OPPOSE_5PT_LABELS),
        "V241181":  ("separate_families_at_border", FAVOR_OPPOSE_5PT_LABELS),
        "V241183":  ("english_official_language", FAVOR_OPPOSE_LABELS),
        "V241184":  ("birthright_citizenship", FAVOR_OPPOSE_LABELS),
        "V241178":  ("pathway_to_citizenship", FAVOR_OPPOSE_5PT_LABELS),
        "V241180":  ("daca_program", FAVOR_OPPOSE_5PT_LABELS),
        "V241182":  ("reduce_legal_immigration", FAVOR_OPPOSE_5PT_LABELS),
        "V241215":  ("undocumented_immigrants_policy", RACIAL_EQUALITY_LABELS),
        # Criminal justice and guns
        "V241219":  ("police_funding_view", POLICE_FUNDING_LABELS),
        "V241253":  ("require_gun_permit", GUN_PERMIT_LABELS),
        # Transgender policy
        "V241218x": ("transgender_policy_view", TRANSGENDER_POLICY_LABELS),
        # Affirmative action
        "V241220":  ("affirmative_action_view", AFFIRM_ACTION_LABELS),
    },
    "racial_attitudes": {
        # Feelings toward racial/ethnic groups
        "V241025":  ("feeling_whites", THERMO),
        "V241027":  ("feeling_blacks", THERMO),
        "V241028":  ("feeling_hispanics", THERMO),
        "V241029":  ("feeling_asian_americans", THERMO),
        "V241035":  ("feeling_illegal_immigrants", THERMO),
        "V241036":  ("feeling_legal_immigrants", THERMO),
        "V241039":  ("feeling_palestinians", THERMO),
        "V241040":  ("feeling_israelis", THERMO),
        # Racial discrimination perceptions
        "V241200":  ("discrimination_whites_jobs", DISCRIM_AMOUNT_LABELS),
        "V241201":  ("discrimination_hispanics_jobs", DISCRIM_AMOUNT_LABELS),
        "V241202":  ("discrimination_blacks_jobs", DISCRIM_AMOUNT_LABELS),
        "V241203":  ("discrimination_asian_americans_jobs", DISCRIM_AMOUNT_LABELS),
        "V241204":  ("police_treatment_race", POLICE_TREAT_LABELS),
        "V241205":  ("influence_slavery_black_position", RACE_INFLUENCE_HIRING_LABELS),
        "V241206":  ("influence_discrimination_black_position", RACE_INFLUENCE_HIRING_LABELS),
        "V241207":  ("influence_lack_motivation_black_position", RACE_INFLUENCE_HIRING_LABELS),
        "V241208":  ("influence_educational_opportunities_black_position", RACE_INFLUENCE_HIRING_LABELS),
        # Racial equality views
        "V241235":  ("racial_equality_attention", RACIAL_EQUALITY_LABELS),
        "V241237":  ("racial_resentment_deserve_more", RACIAL_RESENTMENT_DESERVE_LABELS),
        # Discrimination against specific groups
        "V241241":  ("discrimination_whites_amount", DISCRIM_AMOUNT_LABELS),
    },
    "values_personality": {
        # Moral foundations (0-100 relevance)
        "V241159":  ("moral_foundation_harm", THERMO),
        "V241160":  ("moral_foundation_fairness", THERMO),
        "V241161":  ("moral_foundation_loyalty", THERMO),
        "V241162":  ("moral_foundation_authority", THERMO),
        "V241163":  ("moral_foundation_purity", THERMO),
        # Authoritarianism
        "V241164":  ("auth_independence_vs_respect", AUTH_INDEPENDENCE_LABELS),
        "V241165":  ("auth_curiosity_vs_manners", AUTH_CURIOSITY_LABELS),
        "V241166":  ("auth_selfreliance_vs_obedience", AUTH_SELFRELIANCE_LABELS),
        "V241167":  ("auth_considerate_vs_wellbehaved", AUTH_CONSIDERATE_LABELS),
        # Science attitudes
        "V241169":  ("science_benefits_vs_harms", SCIENCE_GOOD_HARM_LABELS),
        "V241170":  ("science_regulation_view", SCIENCE_REGULATE_LABELS),
        "V241171":  ("trust_in_scientific_community", SCIENCE_TRUST_LABELS),
        # Environment vs economy
        "V241120":  ("environment_vs_jobs_tradeoff", ENVIRO_JOBS_7PT_LABELS),
        # Egalitarianism
        "V241173":  ("worry_less_about_equality", YES_NO_LABELS),
    },
    "media_consumption": {
        # News sources (binary: use/don't use)
        "V241600a": ("news_source_tv", BINARY_01_LABELS),
        "V241600b": ("news_source_newspaper", BINARY_01_LABELS),
        "V241600c": ("news_source_radio", BINARY_01_LABELS),
        "V241600d": ("news_source_internet", BINARY_01_LABELS),
        "V241600e": ("news_source_social_media", BINARY_01_LABELS),
        # Social media and political information
        "V241607":  ("social_media_frequency", SOCIAL_MEDIA_FREQ_LABELS),
        "V241608":  ("political_news_interest", POLITICAL_NEWS_INTEREST_LABELS),
        "V241523":  ("social_media_political_posts_seen", YES_NO_LABELS),
        "V241530":  ("bought_boycotted_product_political", YES_NO_LABELS),
        "V241582x": ("social_media_daily_hours", SOCIAL_MEDIA_HOURS_LABELS),
        # Specific news sources
        "V241620":  ("fox_news_use", FOX_CNN_FREQ_LABELS),
        "V241621":  ("cnn_use", FOX_CNN_FREQ_LABELS),
        # Feelings about institutions
        "V241017":  ("feeling_scientists", THERMO),
        "V241043":  ("feeling_news_media", THERMO),
        "V241015":  ("feeling_police", THERMO),
        "V241016":  ("feeling_military", THERMO),
    },
    "religion_community": {
        # Religion details
        "V241422":  ("religious_tradition", RELIGION_LABELS),
        "V241423":  ("church_attendance", RELIGION_ATTENDANCE_LABELS),
        "V241433":  ("religion_importance_in_life", RELIGION_IMPORTANCE_LABELS),
        "V241434":  ("religion_provides_guidance", RELIGION_GUIDANCE_LABELS),
        # Community and social connections
        "V241470":  ("number_of_children_in_home", CHILDREN_IN_HOME_LABELS),
        "V241471":  ("has_children_under_18", CHILDREN_BINARY_LABELS),
        "V241479":  ("has_internet_access", YES_NO_LABELS),
        "V241482":  ("length_at_current_address_years", None),
        "V241483":  ("born_in_us", YES_NO_LABELS),
        "V241583":  ("loneliness_score", None),
    },
    "policy_positions": {
        # Government spending priorities (favor increase/decrease)
        "V241250": ("spending_social_security", SPENDING_PRIORITY_LABELS),
        "V241251": ("spending_public_schools", SPENDING_PRIORITY_LABELS),
        "V241256": ("spending_science_technology", SPENDING_PRIORITY_LABELS),
        "V241257": ("spending_dealing_with_crime", SPENDING_PRIORITY_LABELS),
        "V241259": ("spending_childcare", SPENDING_PRIORITY_LABELS),
        "V241260": ("spending_aid_to_poor", SPENDING_PRIORITY_LABELS),
        "V241261": ("spending_environment", SPENDING_PRIORITY_LABELS),
        "V241264": ("spending_border_security", SPENDING_PRIORITY_LABELS),
        "V241267": ("spending_highways_infrastructure", SPENDING_PRIORITY_LABELS),
        # Candidate issue placements
        "V241272x": ("harris_ideology_placement", FAVORABILITY_4PT_LABELS),
        "V241273":  ("dem_party_ideology_placement", RACIAL_EQUALITY_LABELS),
        # More specific policy views
        "V241248":  ("abortion_always_vs_never_4pt", POLICE_FUNDING_LABELS),
        "V241134":  ("trump_keep_us_out_of_war", YES_NO_LABELS),
        "V241138":  ("harris_handle_economy", YES_NO_LABELS),
        "V241141":  ("trump_handle_immigration", YES_NO_LABELS),
        "V241144":  ("harris_handle_healthcare", YES_NO_LABELS),
        "V241147":  ("trump_handle_foreign_policy", YES_NO_LABELS),
        "V241150":  ("harris_handle_covid_aftermath", YES_NO_LABELS),
        "V241153":  ("trump_handle_crime", YES_NO_LABELS),
    },
    "civic_participation": {
        "V241044":  ("feeling_united_states", THERMO),
        "V241049":  ("feeling_united_nations", THERMO),
        "V241522":  ("wore_campaign_button", YES_NO_LABELS),
        "V241525":  ("worked_for_party_candidate", YES_NO_LABELS),
        "V241529":  ("displayed_campaign_sign", YES_NO_LABELS),
    },
}


# ---------------------------------------------------------------------------
# Additional label maps for expanded eval items
# ---------------------------------------------------------------------------

NATL_ECON_RETRO_LABELS = {
    "1": "Gotten better", "2": "Stayed about the same", "5": "Gotten worse",
}

PERSONAL_FIN_LABELS = {"1": "Better off", "2": "Worse off"}

GOVT_SERVICES_LABELS = {
    "1": "Government should provide fewer services and reduce spending",
    "2": "Government should provide more services and increase spending",
}

ACA_LABELS = {
    "1": "Favor a great deal", "2": "Favor somewhat",
    "3": "Neither favor nor oppose",
    "4": "Oppose somewhat", "5": "Oppose a great deal",
}

GOVT_PRIV_HEALTH_LABELS = {
    "1": "Government plan",
    "2": "Private insurance",
}

EFFICACY_5PT_LABELS = {
    "1": "Agree strongly", "2": "Agree somewhat",
    "3": "Neither agree nor disagree",
    "4": "Disagree somewhat", "5": "Disagree strongly",
}

AID_BLACKS_7PT_LABELS = {
    "1": "Government should help blacks",
    "2": "2", "3": "3", "4": "4", "5": "5", "6": "6",
    "7": "Blacks should help themselves",
}

DISCRIM_BLACKS_LABELS = {
    "1": "A great deal", "2": "A lot", "3": "A moderate amount",
    "4": "A little", "5": "None at all",
}

NEWER_LIFESTYLES_7PT_LABELS = {
    "1": "Agree strongly that newer lifestyles are contributing to breakdown",
    "2": "2", "3": "3", "4": "4", "5": "5", "6": "6",
    "7": "Disagree strongly",
}

TRAD_FAMILY_7PT_LABELS = {
    "1": "Agree strongly that more emphasis on traditional family ties needed",
    "2": "2", "3": "3", "4": "4", "5": "5", "6": "6",
    "7": "Disagree strongly",
}

SAMESEX_MARRIAGE_7PT_LABELS = {
    "1": "No legal recognition", "2": "2", "3": "3",
    "4": "Civil unions", "5": "5", "6": "6",
    "7": "Marriage",
}

DEATH_PENALTY_7PT_LABELS = {
    "1": "Favor strongly", "2": "2", "3": "3",
    "4": "4", "5": "5", "6": "6",
    "7": "Oppose strongly",
}

GUN_ACCESS_7PT_LABELS = {
    "1": "A lot easier", "2": "2", "3": "3",
    "4": "Kept about the same", "5": "5", "6": "6",
    "7": "A lot more difficult",
}

ABORTION_7PT_LABELS = {
    "1": "Never permit", "2": "2", "3": "3",
    "4": "4", "5": "5", "6": "6",
    "7": "Always allow by law",
}

ENVIRO_REG_LABELS = {
    "1": "Favor regulation", "2": "Oppose regulation",
}

# ---------------------------------------------------------------------------
# Evaluation items (held out from all modules) -- 22 items across domains
# ---------------------------------------------------------------------------

EVAL_ITEMS = {
    # --- Politics / Trust (5) ---
    "V241229": {
        "label": "Trust in federal government",
        "labels": TRUST_GOVT_LABELS,
        "domain": "politics",
        "expected_skill": "attitude_construction",
        "question": (
            "How often can you trust the federal government in Washington "
            "to do what is right? Always, most of the time, about half the "
            "time, some of the time, or never?"
        ),
    },
    "V241231": {
        "label": "Government run by big interests or for all",
        "labels": BIG_INTERESTS_LABELS,
        "domain": "politics",
        "expected_skill": "direct_attitude",
        "question": (
            "Would you say the government is pretty much run by a few big "
            "interests looking out for themselves or that it is run for the "
            "benefit of all the people?"
        ),
    },
    "V241232": {
        "label": "Government waste of tax money",
        "labels": GOVT_WASTE_LABELS,
        "domain": "politics",
        "expected_skill": "direct_attitude",
        "question": (
            "Do you think that people in government waste a lot of the money "
            "we pay in taxes, waste some of it, or don't waste very much of it?"
        ),
    },
    "V241233": {
        "label": "External efficacy: people like me have no say",
        "labels": EFFICACY_5PT_LABELS,
        "domain": "politics",
        "expected_skill": "attitude_construction",
        "question": (
            "How much do you agree or disagree with the following statement: "
            "'People like me don't have any say about what the government does.' "
            "Do you agree strongly, agree somewhat, neither agree nor disagree, "
            "disagree somewhat, or disagree strongly?"
        ),
    },
    "V241234": {
        "label": "External efficacy: public officials don't care",
        "labels": EFFICACY_5PT_LABELS,
        "domain": "politics",
        "expected_skill": "attitude_construction",
        "question": (
            "How much do you agree or disagree with the following statement: "
            "'Public officials don't care much what people like me think.' "
            "Do you agree strongly, agree somewhat, neither agree nor disagree, "
            "disagree somewhat, or disagree strongly?"
        ),
    },
    # --- Economy (3) ---
    "V241108": {
        "label": "National economy retrospective",
        "labels": NATL_ECON_RETRO_LABELS,
        "domain": "economy",
        "expected_skill": "factual_recall",
        "question": (
            "Would you say that over the past year the nation's economy has "
            "gotten better, stayed about the same, or gotten worse?"
        ),
    },
    "V241113": {
        "label": "Personal finances retrospective",
        "labels": PERSONAL_FIN_LABELS,
        "domain": "economy",
        "expected_skill": "factual_recall",
        "question": (
            "Would you say that you and your family are better off or worse "
            "off financially than you were a year ago?"
        ),
    },
    "V241117": {
        "label": "Government spending and services",
        "labels": GOVT_SERVICES_LABELS,
        "domain": "economy",
        "expected_skill": "attitude_construction",
        "question": (
            "Some people think the government should provide fewer services, "
            "even in areas such as health and education, in order to reduce "
            "spending. Other people feel it is important for the government to "
            "provide more services even if it means an increase in spending. "
            "Which view is closer to yours?"
        ),
    },
    # --- Healthcare (2) ---
    "V241210": {
        "label": "Affordable Care Act favor/oppose",
        "labels": ACA_LABELS,
        "domain": "health",
        "expected_skill": "attitude_construction",
        "question": (
            "Do you favor, oppose, or neither favor nor oppose the health care "
            "reform law passed in 2010, also known as the Affordable Care Act "
            "or Obamacare? And would you say you favor/oppose it a great deal "
            "or somewhat?"
        ),
    },
    "V241212": {
        "label": "Government vs private health insurance",
        "labels": GOVT_PRIV_HEALTH_LABELS,
        "domain": "health",
        "expected_skill": "attitude_construction",
        "question": (
            "There is much concern about the rapid rise in medical and hospital "
            "costs. Some feel there should be a government insurance plan which "
            "would cover all medical and hospital expenses. Others feel that "
            "medical expenses should be paid by individuals through private "
            "insurance plans. Which view is closer to yours?"
        ),
    },
    # --- Immigration (1) ---
    "V241177": {
        "label": "Immigration level preference",
        "labels": IMMIGRATION_7PT_LABELS,
        "domain": "social_context",
        "expected_skill": "attitude_construction",
        "question": (
            "Do you think the number of immigrants from foreign countries "
            "who are permitted to come to the United States to live should "
            "be increased a great deal, increased a moderate amount, "
            "increased a little, kept about the same as it is now, decreased "
            "a little, decreased a moderate amount, or decreased a great deal?"
        ),
    },
    # --- Racial attitudes (4) ---
    "V241236": {
        "label": "Racial resentment: slavery and discrimination legacy",
        "labels": EFFICACY_5PT_LABELS,
        "domain": "racial_attitudes",
        "expected_skill": "attitude_construction",
        "question": (
            "Do you agree strongly, agree somewhat, neither agree nor disagree, "
            "disagree somewhat, or disagree strongly with this statement: "
            "'Generations of slavery and discrimination have created conditions "
            "that make it difficult for Blacks to work their way out of the "
            "lower class.'"
        ),
    },
    "V241238": {
        "label": "Racial resentment: Blacks should try harder",
        "labels": EFFICACY_5PT_LABELS,
        "domain": "racial_attitudes",
        "expected_skill": "direct_attitude",
        "question": (
            "Do you agree strongly, agree somewhat, neither agree nor disagree, "
            "disagree somewhat, or disagree strongly with this statement: "
            "'If Blacks would only try harder, they could be just as well off "
            "as whites.'"
        ),
    },
    "V241240": {
        "label": "Government aid to Blacks",
        "labels": AID_BLACKS_7PT_LABELS,
        "domain": "racial_attitudes",
        "expected_skill": "attitude_construction",
        "question": (
            "Some people feel that the government in Washington should make "
            "every effort to improve the social and economic position of Blacks. "
            "Others feel that the government should not make any special effort "
            "to help Blacks because they should help themselves. Where would you "
            "place yourself on this scale? 1 means the government should help "
            "Blacks, 7 means Blacks should help themselves."
        ),
    },
    "V241243": {
        "label": "Amount of discrimination against Blacks",
        "labels": DISCRIM_BLACKS_LABELS,
        "domain": "racial_attitudes",
        "expected_skill": "direct_attitude",
        "question": (
            "How much discrimination is there in the United States today "
            "against each of the following groups? Blacks. Would you say "
            "a great deal, a lot, a moderate amount, a little, or none at all?"
        ),
    },
    # --- Social issues (5) ---
    "V241244": {
        "label": "Newer lifestyles contributing to breakdown",
        "labels": NEWER_LIFESTYLES_7PT_LABELS,
        "domain": "values_personality",
        "expected_skill": "direct_attitude",
        "question": (
            "Please tell me how much you agree or disagree: 'The newer "
            "lifestyles are contributing to the breakdown of our society.' "
            "Use a 7-point scale where 1 means you agree strongly and "
            "7 means you disagree strongly."
        ),
    },
    "V241246": {
        "label": "Emphasis on traditional family ties",
        "labels": TRAD_FAMILY_7PT_LABELS,
        "domain": "values_personality",
        "expected_skill": "direct_attitude",
        "question": (
            "Please tell me how much you agree or disagree: 'This country "
            "would have many fewer problems if there were more emphasis on "
            "traditional family ties.' Use a 7-point scale where 1 means "
            "you agree strongly and 7 means you disagree strongly."
        ),
    },
    "V241247": {
        "label": "Same-sex marriage",
        "labels": SAMESEX_MARRIAGE_7PT_LABELS,
        "domain": "social_context",
        "expected_skill": "direct_attitude",
        "question": (
            "Which comes closest to your view on same-sex couples? "
            "Use a 7-point scale where 1 means there should be no legal "
            "recognition, 4 means civil unions but not marriage, and "
            "7 means same-sex couples should be allowed to marry."
        ),
    },
    "V241252": {
        "label": "Death penalty",
        "labels": DEATH_PENALTY_7PT_LABELS,
        "domain": "social_context",
        "expected_skill": "direct_attitude",
        "question": (
            "Do you favor or oppose the death penalty for persons convicted "
            "of murder? Use a 7-point scale where 1 means you favor strongly "
            "and 7 means you oppose strongly."
        ),
    },
    "V241254": {
        "label": "Gun access",
        "labels": GUN_ACCESS_7PT_LABELS,
        "domain": "social_context",
        "expected_skill": "direct_attitude",
        "question": (
            "Do you think the federal government should make it more difficult "
            "for people to buy a gun, easier, or keep the rules about the same? "
            "Use a 7-point scale where 1 means a lot easier, 4 means kept "
            "about the same, and 7 means a lot more difficult."
        ),
    },
    # --- Abortion (1) ---
    "V241258": {
        "label": "Abortion policy",
        "labels": ABORTION_7PT_LABELS,
        "domain": "social_context",
        "expected_skill": "attitude_construction",
        "question": (
            "Which one of the opinions on this page best agrees with your view "
            "about abortion? Use a 7-point scale where 1 means by law abortion "
            "should never be permitted and 7 means by law a woman should always "
            "be able to obtain an abortion as a matter of personal choice."
        ),
    },
    # --- Environment (1) ---
    "V241175": {
        "label": "Environmental regulation",
        "labels": ENVIRO_REG_LABELS,
        "domain": "values_personality",
        "expected_skill": "attitude_construction",
        "question": (
            "Do you favor or oppose the U.S. government regulating greenhouse "
            "gas emissions from power plants, factories, and cars in an effort "
            "to reduce global warming?"
        ),
    },
}


def safe_lookup(val, labels, default="Not reported"):
    val = str(val).strip()
    return labels.get(val, default)


def is_valid(val):
    v = str(val).strip()
    return v and not v.startswith("-") and v not in ("", " ")


def load_anes():
    if not ANES_CSV.exists():
        print(f"ANES CSV not found: {ANES_CSV}")
        print("Download from: https://electionstudies.org/data-center/")
        sys.exit(1)
    with open(ANES_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def select_respondents(rows, n, seed):
    rng = random.Random(seed)
    complete = [r for r in rows
                if all(is_valid(r.get(v, "")) for v in REQUIRED_VARS)]
    print(f"Respondents with complete demographics: {len(complete)} / {len(rows)}")

    if len(complete) <= n:
        return complete

    strata = {}
    for row in complete:
        pid = row["V241227x"].strip()
        region = row["V243007"].strip()
        strata.setdefault(f"{pid}_{region}", []).append(row)

    selected = []
    keys = sorted(strata.keys())
    rng.shuffle(keys)
    per_stratum = max(1, n // len(keys))
    for key in keys:
        pool = strata[key]
        rng.shuffle(pool)
        selected.extend(pool[:per_stratum])
        if len(selected) >= n:
            break

    if len(selected) < n:
        remaining = [r for r in complete if r not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])

    return selected[:n]


def build_persona(row, persona_id):
    age = row["V241458x"].strip()
    sex = safe_lookup(row["V241550"], GENDER_LABELS)
    race = safe_lookup(row["V241501x"], RACE_LABELS)
    education = safe_lookup(row["V241461x"], EDUCATION_LABELS)
    income = safe_lookup(row["V241567x"], INCOME_6_LABELS)
    religion = safe_lookup(row["V241422"], RELIGION_LABELS)
    region = safe_lookup(row["V243007"], REGION_LABELS)
    marital = safe_lookup(row["V241465x"], MARITAL_LABELS)
    state_code = row.get("V243002", "").strip()
    state = FIPS_STATE.get(state_code, region + " region")

    persona = {
        "demographics": {
            "persona_id": persona_id,
            "age": int(age) if age.isdigit() else age,
            "gender": sex,
            "race": race,
            "education": education,
            "income_bracket": income,
            "marital_status": marital,
            "religion": religion,
            "state": state,
            "region": region,
        },
        "life_narrative": {
            "summary": (
                f"A {age}-year-old {sex.lower()}, {race.lower()}, "
                f"living in {state} ({region}). "
                f"{education}, {marital.lower()}, {religion.lower()}. "
                f"Household income: {income}."
            ),
        },
        "politics": {},
        "economy": {},
        "health": {},
        "social_context": {},
        "racial_attitudes": {},
        "values_personality": {},
        "media_consumption": {},
        "religion_community": {},
        "local_context": {"state": state, "region": region},
        "policy_positions": {},
        "civic_participation": {},
    }

    for module, var_map in MODULE_ITEMS.items():
        for var, (field, labels) in var_map.items():
            val = row.get(var, "").strip()
            if not is_valid(val):
                continue

            if labels == THERMO:
                try:
                    n = int(val)
                    if 0 <= n <= 100:
                        persona[module][field] = thermo_label(n)
                except ValueError:
                    pass
            elif labels is None:
                persona[module][field] = val
            else:
                mapped = safe_lookup(val, labels, None)
                if mapped is not None:
                    persona[module][field] = mapped

    return persona


def save_eval_items(rows, selected_rows, out_path):
    eval_out = []
    for var, info in EVAL_ITEMS.items():
        item = {
            "variable": var,
            "label": info["label"],
            "question_text": info["question"],
            "response_options": info["labels"],
            "domain": info.get("domain", ""),
            "expected_skill": info.get("expected_skill", ""),
            "ground_truth": {},
        }
        for row in selected_rows:
            pid = row["_persona_id"]
            val = row.get(var, "").strip()
            if is_valid(val):
                label = safe_lookup(val, info["labels"], None)
                if label is not None:
                    item["ground_truth"][pid] = {"code": val, "label": label}
        eval_out.append(item)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eval_out, f, indent=2, ensure_ascii=False)
    return eval_out


def main():
    parser = argparse.ArgumentParser(description="Generate personas from ANES 2024")
    parser.add_argument("--n", type=int, default=10, help="Number of personas")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    args = parser.parse_args()

    rows = load_anes()
    print(f"Loaded {len(rows)} ANES 2024 respondents")

    selected = select_respondents(rows, args.n, args.seed)
    print(f"Selected {len(selected)} respondents\n")

    PERSONAS_DIR.mkdir(exist_ok=True)
    for old in PERSONAS_DIR.glob("anes_*.json"):
        old.unlink()

    for i, row in enumerate(selected, 1):
        pid = f"anes_{i:03d}"
        row["_persona_id"] = pid
        persona = build_persona(row, pid)
        out_path = PERSONAS_DIR / f"{pid}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(persona, f, indent=2, ensure_ascii=False)

        d = persona["demographics"]
        pol = persona["politics"]
        total_fields = sum(
            len(v) if isinstance(v, dict) else 1
            for v in persona.values()
        )
        print(f"  {pid}: {d['age']}yo {d['gender']}, {d['race']}, "
              f"{d['education']}, {d['region']}, "
              f"PID={pol.get('party_identification', '?')}, "
              f"fields={total_fields}")

    eval_path = DEMO_DIR / "eval_items.json"
    eval_items = save_eval_items(rows, selected, eval_path)
    total_gt = sum(len(e["ground_truth"]) for e in eval_items)
    print(f"\n{len(eval_items)} eval items with {total_gt} ground-truth responses")
    print(f"Saved to {eval_path}")

    print("\n=== Demographic distribution ===")
    for field in ("gender", "race", "region"):
        vals = []
        for fp in sorted(PERSONAS_DIR.glob("anes_*.json")):
            with open(fp) as f:
                vals.append(json.load(f)["demographics"][field])
        print(f"  {field}: {dict(Counter(vals))}")

    print("\n=== Module richness ===")
    for fp in sorted(PERSONAS_DIR.glob("anes_*.json"))[:3]:
        with open(fp) as f:
            p = json.load(f)
        pid = p["demographics"]["persona_id"]
        total = 0
        parts = []
        for mod, data in p.items():
            n = len(data) if isinstance(data, dict) else 1
            total += n
            parts.append(f"{mod}={n}")
        print(f"  {pid} ({total} total): {', '.join(parts)}")

    print(f"\nDone. {len(selected)} personas in {PERSONAS_DIR}/")


if __name__ == "__main__":
    sys.exit(main())
