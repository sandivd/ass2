import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


PROCESSED_DATA_PATH = 'final_processed_data.csv'

# load processed data
merged_df = pd.read_csv(PROCESSED_DATA_PATH)

# visualization setup
sns.set_theme(style="whitegrid")
plt.style.use('seaborn-v0_8-colorblind')


# Visualization 1: Overall Severity Distribution
plt.figure(figsize=(8, 6))
severity_order_plot = ['1: Fatal accident', '2: Serious injury accident', '3: Other injury accident', '4: Non injury accident', 'Unknown']
existing_severities = [s for s in severity_order_plot if s in merged_df['SEVERITY_DESC'].unique()]

accident_level_df_vis1 = merged_df.drop_duplicates(subset=['ACCIDENT_NO'])
severity_counts = accident_level_df_vis1['SEVERITY_DESC'].value_counts().reindex(existing_severities).fillna(0)

sns.barplot(x=severity_counts.index, y=severity_counts.values, order=existing_severities)
plt.title('Distribution of Accident Severity Levels (Unique Accidents)')
plt.xlabel('Severity Level')
plt.ylabel('Number of Unique Accidents')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('vis1_severity_distribution.png')
plt.close()

# Visualization 2: Severity vs. Speed Zone
plt.figure(figsize=(12, 7))
accident_level_df_vis2 = merged_df.drop_duplicates(subset=['ACCIDENT_NO'])
speed_zone_order = sorted(accident_level_df_vis2['SPEED_ZONE_GROUP'].dropna().unique(), key=lambda x: (x != 'low speed', x != 'medium speed', x != 'high speed', x))

ax = sns.histplot(data=accident_level_df_vis2.dropna(subset=['SPEED_ZONE_GROUP', 'SEVERITY_DESC']),
             x='SPEED_ZONE_GROUP', hue='SEVERITY_DESC',
             multiple='fill', stat='proportion', shrink=0.8, common_norm=False,
             hue_order=existing_severities, discrete=True, ) # Help from Gemini 2.5 Pro obtained for the parameters
plt.gca().set_xticklabels(speed_zone_order) # Help from Claude 3.7 obtained for this line
plt.title('Proportion of Accident Severity Levels by Speed Zone Group')
plt.xlabel('Speed Zone Group')
plt.ylabel('Proportion of Accidents')
plt.xticks(rotation=45, ha='right')
legend = ax.get_legend() # Get the legend object created by seaborn
if legend: # Check if a legend exists
    legend.set_bbox_to_anchor((1.02, 1)) # Adjust the legend position
plt.tight_layout()
plt.savefig('vis2_severity_vs_speedzone.png', bbox_inches='tight')
plt.close()

# Visualization 3: Severity vs. Vehicle Age Group
plt.figure(figsize=(12, 7))
vehicle_id_col_name = 'VEHICLE_ID_x'

vehicle_level_df_vis3 = merged_df.drop_duplicates(subset=['ACCIDENT_NO', vehicle_id_col_name])
vehicle_age_order = ['0-4 years', '5-9 years', '10-14 years', '15+ years', 'Unknown']
vehicle_age_order_present = [v for v in vehicle_age_order if v in vehicle_level_df_vis3['VEHICLE_AGE_GROUP'].unique()]

ax = sns.histplot(data=vehicle_level_df_vis3.dropna(subset=['VEHICLE_AGE_GROUP', 'SEVERITY_DESC']),
             x='VEHICLE_AGE_GROUP', hue='SEVERITY_DESC',
             multiple='fill', stat='proportion', shrink=0.8, common_norm=False,
             hue_order=existing_severities, discrete=True) # Help from Gemini 2.5 Pro obtained for the parameters
plt.gca().set_xticklabels(vehicle_age_order_present) # Help from Claude 3.7 obtained for this line
plt.title('Proportion of Accident Severity by Vehicle Age Group (Unique Vehicles)')
plt.xlabel('Vehicle Age Group')
plt.ylabel('Proportion of Vehicles Involved')
legend = ax.get_legend() # Get the legend object created by seaborn
if legend: # Check if a legend exists
    legend.set_bbox_to_anchor((1.02, 1)) # Adjust the legend position
plt.tight_layout()
plt.savefig('vis3_severity_vs_speedzone.png', bbox_inches='tight')
plt.close()

# Visualization 4: Severity vs. Road User Type
plt.figure(figsize=(12, 7))
road_user_order = ['Driver', 'Passenger', 'Pedestrian', 'Cyclist', 'Unknown']
road_user_order_present = [r for r in road_user_order if r in merged_df['ROAD_USER_TYPE_DESC'].unique()]
ax = sns.histplot(data=merged_df.dropna(subset=['ROAD_USER_TYPE_DESC', 'SEVERITY_DESC']),
             x='ROAD_USER_TYPE_DESC', hue='SEVERITY_DESC',
             multiple='fill', stat='proportion', shrink=0.8, common_norm=False,
             hue_order=existing_severities, discrete=True) # Help from Gemini 2.5 Pro obtained for the parameters
plt.gca().set_xticklabels(road_user_order_present) # Help from Claude 3.7 obtained for this line
plt.title('Proportion of Injury Severity by Road User Type')
plt.xlabel('Road User Type')
plt.ylabel('Proportion of Persons Involved')
plt.xticks(rotation=45, ha='right')
legend = ax.get_legend() # Get the legend object created by seaborn
if legend: # Check if a legend exists
    legend.set_bbox_to_anchor((1.02, 1)) # Adjust the legend position
plt.tight_layout()
plt.savefig('vis4_severity_vs_speedzone.png', bbox_inches='tight')
plt.close()

# Visualization 5: Severity vs. Light Condition
plt.figure(figsize=(12, 7))
accident_level_df_vis5 = merged_df.drop_duplicates(subset=['ACCIDENT_NO'])
light_cond_order = ['daytime', 'dusk', 'nighttime（light on）', 'nighttime（no light）', 'unknown']
light_cond_order_present = [l for l in light_cond_order if l in accident_level_df_vis5['LIGHT_CONDITION_DESC'].unique()]

ax = sns.histplot(data=accident_level_df_vis5.dropna(subset=['LIGHT_CONDITION_DESC', 'SEVERITY_DESC']),
             x='LIGHT_CONDITION_DESC', hue='SEVERITY_DESC',
             multiple='fill', stat='proportion', shrink=0.8, common_norm=False,
             hue_order=existing_severities, discrete=True) # Help from Gemini 2.5 Pro obtained for the parameters
plt.gca().set_xticklabels(light_cond_order_present) # Help from Claude 3.7 obtained for this line
plt.title('Proportion of Accident Severity by Light Condition')
plt.xlabel('Light Condition')
plt.ylabel('Proportion of Accidents')
plt.xticks(rotation=45, ha='right')
legend = ax.get_legend() # Get the legend object created by seaborn
if legend: # Check if a legend exists
    legend.set_bbox_to_anchor((1.02, 1)) # Adjust the legend position
plt.tight_layout()
plt.savefig('vis5_severity_vs_speedzone.png', bbox_inches='tight')
plt.close()

print("\nAll planned visualizations generated and saved.")