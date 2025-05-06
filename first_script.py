import pandas as pd
import glob
import re
from sklearn.preprocessing import LabelEncoder
import numpy as np

csv_files = glob.glob("*.csv")

df_accident = pd.read_csv('accident.csv')
df_vehicle = pd.read_csv('vehicle.csv')
df_person = pd.read_csv('person.csv')
keep_column_accident = ['ACCIDENT_NO', 'ACCIDENT_DATE', 'ACCIDENT_TIME',
    'ACCIDENT_TYPE', 'ACCIDENT_TYPE_DESC', 'DAY_OF_WEEK',
    'LIGHT_CONDITION', 'NO_OF_VEHICLES',
    'NO_PERSONS_KILLED', 'NO_PERSONS_INJ_2', 'NO_PERSONS_INJ_3',
    'NO_PERSONS_NOT_INJ', 'NO_PERSONS',
    'POLICE_ATTEND', 'ROAD_GEOMETRY',
    'SEVERITY', 'SPEED_ZONE']
df_accident = df_accident[keep_column_accident]

#create a column that displays only the year of the accident happened
df_accident['acc_year'] = df_accident['ACCIDENT_DATE'].str.extract(r'(\d{4})').astype(int)


keep_column_vehicle = [
    'ACCIDENT_NO', 'VEHICLE_ID', 'VEHICLE_YEAR_MANUF',
    'ROAD_SURFACE_TYPE', 'ROAD_SURFACE_TYPE_DESC', 'VEHICLE_MAKE',
    'VEHICLE_TYPE', 'VEHICLE_TYPE_DESC', 'LEVEL_OF_DAMAGE',
    'INITIAL_IMPACT', 'DRIVER_INTENT', 'VEHICLE_MOVEMENT',
    'CAUGHT_FIRE', 'TOTAL_NO_OCCUPANTS'
]
df_vehicle = df_vehicle[keep_column_vehicle]


keep_column_person = [
    'ACCIDENT_NO', 'PERSON_ID', 'VEHICLE_ID',
    'SEX', 'AGE_GROUP', 'INJ_LEVEL',
    'SEATING_POSITION', 'HELMET_BELT_WORN',
    'ROAD_USER_TYPE','ROAD_USER_TYPE_DESC', 'TAKEN_HOSPITAL', 'EJECTED_CODE'
]
df_person = df_person[keep_column_person]
#replace all the other sex data except M or F to the mode of sex
mode_sex = df_person['SEX'].mode()[0]
df_person['SEX'] = df_person['SEX'].apply(lambda x: x if x in ['M', 'F'] else mode_sex)
#transform all the age_group to the normalized format, extreme example like 5-MAY will transform to Unknown
#the age group like 70+ will be kept
valid_age_group_regex = r'^\d{1,2}-\d{1,2}$|^\d{2}\+$'
df_person['AGE_GROUP'] = df_person['AGE_GROUP'].where(df_person['AGE_GROUP'].str.match(valid_age_group_regex), 'Unknown')
def categorize_age(age_group):
    if re.match(r'^\d{2}\+$', age_group):  
        base_age = int(age_group[:-1]) 
        if base_age >= 70:
            return '70+'  
    elif re.match(r'^\d{1,2}-\d{1,2}$', age_group):  
        start_age, end_age = map(int, age_group.split('-'))
        if start_age < 18:
            return '0-17'
        elif 18 <= start_age <= 29:
            return '18-29'
        elif 30 <= start_age <= 39:
            return '30-39'
        elif 40 <= start_age <= 49:
            return '40-49'
        elif 50 <= start_age <= 59:
            return '50-59'
        elif 60 <= start_age <= 69:
            return '60-69'
        elif start_age >= 70:
            return '70+'
    return 'Unknown'  

df_person['AGE_GROUP'] = df_person['AGE_GROUP'].apply(categorize_age)
#replace unknown data to the mode of 'AGE_GROUP'
moded_age_group = df_person['AGE_GROUP'].mode()[0]
df_person['AGE_GROUP'] = df_person['AGE_GROUP'].replace('Unknown', moded_age_group)
#fill the unknown helmet worn data with mode data
mode_helmet = df_person['HELMET_BELT_WORN'].mode()[0]
df_person['HELMET_BELT_WORN'] = df_person['HELMET_BELT_WORN'].fillna(mode_helmet)
#reduce the group of road_users
road_usertype_map = {
    'Drivers': 'Driver',
    'Motorcyclists': 'Driver',
    'Pillion Passengers': 'Passenger',
    'Passengers': 'Passenger',
    'Pedestrians': 'Pedestrian',
    'Bicyclists': 'Cyclist',
    'E-scooter Rider': 'Cyclist',
    'Not Known': 'Unknown'
}
df_person['ROAD_USER_TYPE_DESC'] = df_person['ROAD_USER_TYPE_DESC'].map(road_usertype_map)
#re define the road_user_type
le = LabelEncoder()
df_person['ROAD_USER_TYPE'] = le.fit_transform(df_person['ROAD_USER_TYPE_DESC'])

fill_map = (
    df_person[df_person['TAKEN_HOSPITAL'] != 'nah']
    .groupby('INJ_LEVEL')['TAKEN_HOSPITAL']
    .agg(lambda x: x.mode()[0])
    .to_dict()
)
# further correction needed, trying to convert all the empty data with reference
#of a dictionary which contains the mode of different injury levels
#eg. dic(1;Y, 2:Y, 3:N .etc )
df_person['TAKEN_HOSPITAL'] = df_person['TAKEN_HOSPITAL'].replace('', 'N')


# for vehicle data cleaning
# replace all the 0s with missing values
df_vehicle['VEHICLE_YEAR_MANUF'] = df_vehicle['VEHICLE_YEAR_MANUF'].replace(0, np.nan)

# fill rows having missing values with the mode of the given columns

v_modes = df_vehicle.mode().iloc[0]


# we want to calculate the age of vehicles later so we cannot fill na with mode for year of made
cols_to_fill = df_vehicle.columns.difference(['VEHICLE_YEAR_MANUF'])
df_vehicle[cols_to_fill] = df_vehicle[cols_to_fill].fillna(v_modes)


acc_veh = pd.merge(df_accident, df_vehicle, on='ACCIDENT_NO', how='left')

# now get the age of the vehicle
acc_veh['AGE'] = acc_veh['acc_year'] - acc_veh['VEHICLE_YEAR_MANUF']
mean_age = acc_veh['AGE'].mean()
# fill the missing values in age with mean_age because we will make two groups later
# and we don't want one group is way more than the other, and the threshold can have decimal places
acc_veh['AGE'] = acc_veh['AGE'].fillna(mean_age)

merged_df = pd.merge(acc_veh, df_person, on='ACCIDENT_NO', how='left')



