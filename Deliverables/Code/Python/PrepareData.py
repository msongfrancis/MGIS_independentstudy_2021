"""
Preparing E-scooter data

This script joins tabular e-scooter data from different years with demographic data and
spatial census tracts and is intended to compute total trip counts from starting trip
centerlines. The datasets outputted are intended to be used as inputs for machine learning
regressions to predict total trips based on demographic values. 

The script requires the pandas module and ArcGIS pro license.

Data sources: ACS-Survey 2014-2018 5-year Estimates, ACS-Survey 2015-2019 5-year Estimates,
City of Minneapolis, U.S. Census Bureau
"""

import os
import pandas as pd

data_dir = r"C:\Users\msong\Desktop\Independent proj\data\scooter_mpls" #scooter trips
# data_dir = r"C:\Users\msong\Desktop\Independent proj\data\nhgis0020_csv" #census tables
os.chdir(data_dir)

def reformat_scooter_tables(in_table,columns,data_yr):
    """ Computes a table with the data year and the total escooter trips for the
    street centerline and outputs a csv file.
    
    Parameters:
    -----------
    in_table: csv
        The escooter trip table with multiple years of data in one table. 
    columns: list or str
        The fields of interest to be used when reading the in_table
    data_yr: list
        The years for the data present in the in_table
        
    Output
    ------
    csv file: csv
        Table with newly calculated TripIDs and total trip counts for
        each centerline in dataset.
    """
    
    df = pd.read_csv(os.path.join(data_dir, in_table), usecols=columns)
    df["TripID"] = data_yr + df["TripID"].astype(str) # create unique ID with year
    df_count = df.groupby(['StartCenterlineID']).count()
    df["StartCenterlineID"] = df["StartCenterlineID"].astype(str)
    df_count = df_count[['TripID']]
    df_count = df_count.rename(columns = {'TripID':f'TripCount'})
    df_count["year"] = f"{data_yr}"
    df_count.to_csv(f"escooter_tripcount_{data_yr}.csv", index=True)

# Columns of interest    
columns = ["TripID",
           "TripDuration", 
           "TripDistance", 
           "StartTime", 
           "EndTime", 
           "StartCenterlineID", 
           "EndCenterlineID"]
data_yrs = ["2018","2019"]


in_table = f"Motorized_Foot_Scooter_Trips_{data_yrs[0]}.csv"
reformat_scooter_tables(in_table,columns,data_yrs[0])

in_table = f"Motorized_Foot_Scooter_Trips_{data_yrs[1]}.csv"
reformat_scooter_tables(in_table,columns,data_yrs[1])

escooter18 = pd.read_csv(os.path.join(data_dir,"escooter_tripcount_2018.csv"))
escooter19 = pd.read_csv(os.path.join(data_dir,"escooter_tripcount_2019.csv"))

escooter18.head()

escooter18["StartCenterlineID"] = escooter18["StartCenterlineID"].astype(str)
escooter19["StartCenterlineID"] = escooter19["StartCenterlineID"].astype(str)

tripcounts = pd.concat([escooter18,escooter19]) # combine data as one dataframe
tripcounts.to_csv(os.path.join(data_dir,"tripcounts.csv"),index=False)

# import tripcounts into working file geodatabase
arcpy.conversion.TableToTable(f"{os.path.join(data_dir,'tripcounts.csv')}", 
                              arcpy.env.workspace, 
                              "tripcounts")

# determine which census tract each street centerline falls in
arcpy.analysis.SpatialJoin("mpls_streetcenterlines", 
                           "mpls_censustracts_2010", 
                           "mpls_streetcenterlines_sj", 
                           "JOIN_ONE_TO_ONE", 
                           "KEEP_ALL"
                          )

# convert GBSID (joining field) to text for joining
arcpy.management.AddField("mpls_streetcenterlines_sj", 
                          "GBSID_str", 
                          "TEXT")
arcpy.management.CalculateField("mpls_streetcenterlines_sj", 
                                "GBSID_str", 
                                "!GBSID!", 
                                "PYTHON3", 
                                '', 
                                "TEXT")

# convert GISJOINID (joining field) to text for joining
arcpy.management.AddField("tripcounts", 
                          "GISJOINID", 
                          "TEXT")

# join trip counts with street centerlines spatially
arcpy.management.AddJoin("tripcounts", 
                         "StartCenterlineID", 
                         "mpls_streetcenterlines_sj", 
                         "GBSID_str", 
                         "KEEP_COMMON")

# copy field from joining to the spatial features
arcpy.management.CalculateField("tripcounts", 
                                "tripcounts.GISJOINID", 
                                "!mpls_streetcenterlines_sj.GISJOIN!", 
                                "PYTHON3", 
                                '', 
                                "TEXT")

# remove join
arcpy.management.RemoveJoin("tripcounts", 
                            "mpls_streetcenterlines_sj")

# calculate trip counts for each census tract
arcpy.analysis.Statistics("tripcounts", 
                          "tripcounts_stats", 
                          "TripCount SUM", 
                          "year;GISJOINID")

# create a new ID with the year concatenated is obtained from
arcpy.management.AddField("tripcounts_stats", 
                          "joinid_yr", 
                          "TEXT")

codeblock = """
def concatenate(x,y):
    if x == "<Null>":
        x = None
        return x
    else:
        field = f"{x}_{str(y)}"
        return field

"""
arcpy.management.CalculateField("tripcounts_stats", 
                                "joinid_yr", 
                                "concatenate(!GISJOINID!,!year!)", 
                                "PYTHON3", 
                                codeblock,
                                "TEXT")




# data_dir = r"C:\Users\msong\Desktop\Independent proj\data\scooter_mpls" #scooter trips
data_dir = r"C:\Users\msong\Desktop\Independent proj\data\nhgis0020_csv" #census tables
os.chdir(data_dir)

census_2018 = "nhgis0020_ds239_20185_tract.csv"
cols = ["GISJOIN",
        "STATE",
        "STATEA",
        "COUNTY",
        "COUNTYA",
        "TRACTA",
        "AJWNE001", # total pop per tract
        "AJWNE002", # total white pop per tract
        "AJYPE001", # total pop per tract
        "AJYPE017", # highschool
        "AJYPE018", # GED or alt credential
        "AJYPE019", # some college, less than 1 yr
        "AJYPE020", # some college, 1 or more yrs
        "AJYPE021", # associate's degree
        "AJYPE022", # bachelor's degree
        "AJYPE023", # master's degree
        "AJYPE024", # professional school degree
        "AJYPE025", # doctorate degree
        "AJZAE001"  # med hh inc
       ]

c18_df = pd.read_csv(os.path.join(data_dir,census_2018),usecols=cols,encoding='latin-1')

# reduce to state of MN and Hennepin County
h18_df = c18_df.loc[(c18_df["STATEA"]==27) & (c18_df["COUNTYA"]==53)].copy()

# calculate percentage non-white for each tract
h18_df["percent_nonwhite"] = 1 - (h18_df["AJWNE002"]/ h18_df["AJWNE001"])

# calculate population with highschool education or above for each tract
h18_df["percent_hsandabv"] = (h18_df.iloc[0:,9:18].sum(axis=1)) / h18_df["AJWNE001"]

# rescale median household income on a scale of 0-1
# minmax scaling
a, b = 0, 1 
x, y = h18_df.AJZAE001.min(), h18_df.AJZAE001.max()
h18_df["medhhinc_normal"] = (h18_df.AJZAE001 - x) / (y - x) * (b - a) + a
h18_df["year"] = "2018"

h18_df.head()

# drop unneccessary columns 
h18_df = h18_df[["GISJOIN",
                 "year",
                 "AJWNE001",
                "percent_nonwhite",
                "percent_hsandabv",
                "medhhinc_normal",
                "AJZAE001"]]

# rename column
h18_df = h18_df.rename(columns = {"AJZAE001": "med_hh_inc","AJWNE001":"total_pop"})

census_2019 = "nhgis0020_ds244_20195_tract.csv"
columns = ["GISJOIN",
           "STATE",
           "STATEA",
           "COUNTY",
           "COUNTYA",
           "TRACTA",
          "ALUCE001", # total pop per tract
          "ALUCE002", # total white pop per tract
          "ALWGE001", # total pop per tract
          "ALWGE017", # highschool
          "ALWGE018", # GED or alt credential
          "ALWGE019", # some college, less than 1 yr
          "ALWGE020", # some college, 1 or more yrs
          "ALWGE021", # associate's degree
          "ALWGE022", # bachelor's degree
          "ALWGE023", # master's degree
          "ALWGE024", # professional school degree
          "ALWGE025", # doctorate degree
          "ALW1E001"] # med hh inc

c19_df = pd.read_csv(os.path.join(data_dir,census_2019),
                     usecols=columns,
                    encoding='latin-1')

# reduce to state of MN and Hennepin County
h19_df = c19_df.loc[(c19_df["STATEA"]==27) & (c19_df["COUNTYA"]==53)].copy()

# calculate percentage non-white for each tract
h19_df["percent_nonwhite"] = 1 - (h19_df["ALUCE002"]/ h19_df["ALUCE001"])

# calculate population with highschool education or above for each tract
h19_df["percent_hsandabv"] = (h19_df.iloc[0:,9:18].sum(axis=1)) / h19_df["ALWGE001"]

# rescale median household income on a scale of 0-1
# minmax scaling
a, b = 0, 1 
x, y = h19_df.ALW1E001.min(), h19_df.ALW1E001.max()
h19_df["medhhinc_normal"] = (h19_df.ALW1E001 - x) / (y - x) * (b - a) + a
h19_df["year"]="2019"

# drop unneccessary columns 
h19_df = h19_df[["GISJOIN",
                 "year",
                 "ALUCE001",
                "percent_nonwhite",
                "percent_hsandabv",
                "medhhinc_normal",
                "ALW1E001"]]

# rename column
h19_df = h19_df.rename(columns = {"ALW1E001": "med_hh_inc","ALUCE001":"total_pop"})

# merge each year into one dataframe to join with spatial
mdf = pd.concat([h18_df,h19_df])

mdf.to_csv(os.path.join(data_dir,"demographics.csv"),index=False)

arcpy.conversion.TableToTable(os.path.join(data_dir,"demographics.csv"), 
                              arcpy.env.workspace, 
                              "demographics")

# calculate population density
arcpy.management.AddField("demographics", 
                          "popdens_sqmi", 
                          "DOUBLE")

arcpy.management.AddJoin("demographics", 
                         "GISJOIN", 
                         "mpls_censustracts_2010", 
                         "GISJOIN", 
                         "KEEP_ALL")

arcpy.management.CalculateField("demographics", 
                                "demographics.popdens_sqmi", 
                                "!demographics.total_pop!/!mpls_censustracts_2010.area_sqmile!"
                               )
arcpy.management.RemoveJoin("demographics", "mpls_censustracts_2010")

# create a join field for each census tract for each yearly dataset
arcpy.management.AddField("demographics", 
                          "joinid_yrdem", 
                          "TEXT")

codeblock = """
def concatenate(x,y):
    if x == "<Null>":
        x = None
        return x
    else:
        field = f"{x}_{str(y)}"
        return field

"""
arcpy.management.CalculateField("demographics", 
                                "joinid_yrdem", 
                                "concatenate(!GISJOIN!,!year!)", 
                                "PYTHON3", 
                                codeblock,
                                "TEXT")

# join trip counts with demographic data
arcpy.management.AddJoin("tripcounts_stats", 
                         "joinid_yr", 
                         "demographics", 
                         "joinid_yrdem", 
                         "KEEP_ALL")

# import table into working file geodatabase
arcpy.conversion.TableToTable("tripcounts_stats", 
                              r"C:\Users\msong\Desktop\Independent proj\escooter_ML", 
                              "escooter_all.csv", 
                              '', 
                              'GISJOIN "GISJOIN" true true false 8000 Text 0 0,First,#,tripcounts_stats,demographics.GISJOIN,0,8000;year "year" true true false 4 Long 0 0,First,#,tripcounts_stats,tripcounts_stats.year,-1,-1;SUM_TripCount "SUM_TripCount" true true false 8 Double 0 0,First,#,tripcounts_stats,tripcounts_stats.SUM_TripCount,-1,-1;total_pop "total_pop" true true false 4 Long 0 0,First,#,tripcounts_stats,demographics.total_pop,-1,-1;percent_nonwhite "percent_nonwhite" true true false 8 Double 0 0,First,#,tripcounts_stats,demographics.percent_nonwhite,-1,-1;percent_hsandabv "percent_hsandabv" true true false 8 Double 0 0,First,#,tripcounts_stats,demographics.percent_hsandabv,-1,-1;medhhinc_normal "medhhinc_normal" true true false 8 Double 0 0,First,#,tripcounts_stats,demographics.medhhinc_normal,-1,-1;med_hh_inc "med_hh_inc" true true false 8 Double 0 0,First,#,tripcounts_stats,demographics.med_hh_inc,-1,-1;popdens_sqmi "popdens_sqmi" true true false 8 Double 0 0,First,#,tripcounts_stats,demographics.popdens_sqmi,-1,-1',
                              '')


