import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point
import numpy as np
import os
import pickle
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import pickle

def get_response(address):
    address = address.replace(' ', '+').replace('#', '')
    # Create your own Google API key
    # https://developers.google.com/maps/documentation/javascript/get-api-key
    google_api_key = ""
    url = f'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={google_api_key}'
    print(url)
    response = requests.get(url).json()
    return response

def extract_colorado(response):
    for result in response['results']:
        for component in result['address_components']:
            if component['long_name'] == 'Colorado':
                return result

def add_geometry(data, dataname):    
    unique_addresses = data['Combined Address'].unique()
    filename = f'rorco_sources/address_to_location.dict'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            address_to_location = pickle.load(f)
    else:
        address_to_location = {}
    unique_locations = []
    for unique_address in unique_addresses:
        if unique_address not in address_to_location:
            response = get_response(unique_address)
            response_colorado = extract_colorado(response)
            location = response_colorado['geometry']['location']
            address_to_location[unique_address] = location
        unique_locations += [address_to_location[unique_address]]
    with open(filename, 'wb') as f:
        pickle.dump(address_to_location, f)

    unique_data = gpd.GeoDataFrame({
    'Combined Address' : unique_addresses,
    'Latitude' : [location['lat'] for location in unique_locations],
    'Longitude' : [location['lng'] for location in unique_locations],
    'geometry' : [Point(location['lng'],location['lat']) for location in unique_locations]
    }, geometry='geometry', crs='WGS84')

    data_geom = data.merge(unique_data, on='Combined Address')
    if 'geometry_x' in data_geom.columns:
        data_geom.drop(columns=['geometry_x'], inplace=True)
        data_geom.rename(columns={'geometry_y':'geometry'}, inplace=True)
        data_geom = data_geom.set_geometry('geometry')
    return data_geom

def extract_school_number(school):
    last_paren = school.rindex('(')
    remaining = school[last_paren-1:]
    return int(remaining[remaining.index('(')+1:remaining.index(')')])

def extract_district_number(school):
    school = school.replace('(J)', '')
    return int(school[school.index('(')+1:school.index(')')])

def add_school_data(school_geom):
    # Free and reduced lunch data
    filename_frl = 'rorco_sources/2022-23_K12_FRL_bySchool - Suppressed.xlsx'
    # Skip first two rows, only read in first 1934 rows (the rest are empty or not schools)
    school_frl = pd.read_excel(filename_frl, header=2, nrows=1934)
    school_frl.rename(columns={ 'SCHOOL CODE' : 'School Code'}, inplace=True)
    school_frl = school_frl.replace('*', 0) # They coded 0 as *
    # IPST data
    filename_ipst = 'rorco_sources/2022-23_IPST_bySchool - Suppressed.xlsx'
    # Skip first row, only read in first 1934 rows (the rest are empty or not schools)
    school_ipst = pd.read_excel(filename_ipst, header=1, nrows=1934)
    school_ipst = school_ipst.replace('*', 0) # They coded 0 as *

    # Demographic data
    filename_membership = 'rorco_sources/2022-23_Membership_Grade_bySchool.xlsx'
    school_membership = pd.read_excel(filename_membership, header=2, nrows=1934)
    school_membership.rename(columns={'Sch Code':'School Code'}, inplace=True)

    # Attendance data
    filename_attendance = 'rorco_sources/2019-20_Attendance and Truancy Rates by School.xlsx'

    school_attendance = pd.read_excel(filename_attendance, header=6, nrows=2079)
    school_attendance = school_attendance[school_attendance['School Code'] != 'Total']
    school_attendance['School Code'] = school_attendance['School Code'].astype(int)

    # Student teacher ratio
    filename_ratio = 'rorco_sources/2022.2023PupilTeacherRatio.publishedreport.xlsx'

    school_ratio = pd.read_excel(filename_ratio, header=2, nrows=1913)

    # Salary by district
    filename_salary = 'rorco_sources/2022.2023AverageTeachersSalary.FTE_.PublishedReport.updated4.13.23.xlsx'

    district_salary = pd.read_excel(filename_salary, header=3, nrows=196)

    district_salary.rename(columns={'Unnamed: 0':'District Code', 'Unnamed: 1' : 'District'}, inplace=True)

    school_data_sets = [school_frl, school_ipst, school_membership, school_attendance, school_ratio]

    merged = school_geom
    for data in school_data_sets:
        merged = pd.merge(merged, data, on='School Code')
        to_rename = {col : col[-2:] for col in merged.columns if '_x' in col}
        merged.rename(columns=to_rename, inplace=True)
        to_drop = [col for col in merged.columns if '_y' in col]
        merged.drop(columns=to_drop, inplace=True)

    # Get number of each district
    merged['District Code'] = merged['State/District/School'].apply(extract_district_number)

    district_data_sets = [district_salary]
    for data in district_data_sets:
        merged = pd.merge(merged, data, on='District Code')

    return merged 

def match_arrays(array1, array2, verbose=False):
    # Find most similar items in array2 to items in array1
    # Return dictionary where keys are in array1 and values are in array2
    array = {}
    for object1 in set(array1):
        best_ratio = 0
        best_match = None
        for object2 in set(array2):
            ratio = SequenceMatcher(None, object1, object2).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = object2
        if verbose:
            print(object1)
            print(best_match)
        array[object1] = best_match
    return array

def load_school_data():
    test_years = ['2014-2015', '2015-2016', '2016-2017', '2017-2018', '2018-2019']
    suffixes = ['A', 'BC', 'DEF', 'GHIJKL', 'MNOPQR', 'STUVWXYZ']
    list_data = []
    for test_year in test_years:
        for suffix in suffixes:
            filename_school = f'rorco_sources/State Assessment Data Lab_{test_year}_{suffix}.xlsx'
            data = pd.read_excel(filename_school, header=18)
            list_data += [data]

    data = pd.concat(list_data, ignore_index=True, sort=False)
    school_data = gpd.GeoDataFrame(data)

    # Convert empty strings to NaN
    school_data = school_data.replace(r'^\s*$', np.nan, regex=True)
    # Fill in missing values with previous value
    for col in ['Academic Year', 'State/District/School', 'Test Name', 'Subject']:
        school_data[col] = school_data[col].ffill()

    school_data = school_data[school_data['Mean Scale Score'] != '-']

    # Get unique identifier for each school
    school_data['School Code'] = school_data['State/District/School'].apply(extract_school_number)

    # Add addresses
    filename_school_addresses = 'rorco_sources/Public School Mailing Labels 2023-2024.csv'
    school_addresses = pd.read_csv(filename_school_addresses)

    # Ignore schools that don't have a physical address
    school_data = pd.merge(school_data, school_addresses, on='School Code')

    school_data['Combined Address'] = school_data['Physical Address'] + ', ' + school_data['City'] + ', CO'

    school_geom = add_geometry(school_data, 'schools')
    school_geom = add_school_data(school_geom)

    # Create unique index
    triple_index = []
    for year, grade, school_code in zip(school_geom['Academic Year'], school_geom['Grade'], school_geom['School Code']):
        triple_index += [year + '_' + grade + '_' + str(school_code)]

    school_geom['triple_index'] = triple_index
    school_geom.set_index('triple_index', inplace=True)

    # Convert to geodataframe
    school_geom = gpd.GeoDataFrame(school_geom)
    return school_geom

def load_clinic_data():
    # Read in school district data
    district_boundaries = gpd.read_file('rorco_sources/CDPHE_CDOE_School_District_Boundaries/CDPHE_CDOE_School_District_Boundaries.shp')
    # Read in the clinic data
    filename = 'rorco_sources/rorco_cumulative.csv'
    clinics_data = gpd.read_file(filename)
    # Convert columns with numbers to numeric
    columns_with_numeric = [col for col in clinics_data.columns if 'Books' in col or 'Visits' in col]
    for col in columns_with_numeric:
        clinics_data[col] = pd.to_numeric(clinics_data[col])
    # Data only has addresses, add geometry with lat/long from Google API
    clinics_data['Combined Address'] = clinics_data['Physical Location: Line 1'] + ', ' + clinics_data['Physical Location: City'] + ', CO'
    clinics_geom = add_geometry(clinics_data, 'clinics') 
    clinics_geom = gpd.sjoin(clinics_geom, district_boundaries)
    clinics_geom.rename(columns={'index_right': 'District Code', 'NAME' : 'District Name'}, inplace=True)
    return clinics_geom

def assign_visits_to_schools(school_geom, clinics_geom):
    # Want row with year, number of well child visits, and geometry
    # Group by year and geometry, sum well child visits
    clinics = gpd.GeoDataFrame(clinics_geom.groupby(['Report Submission Year', 'geometry']).sum()['Well-Child Visits 6 mos - 5 yrs']).reset_index()
    clinics.set_geometry('geometry', inplace=True)
    clinics.rename(columns={'Report Submission Year' : 'Year', 'Well-Child Visits 6 mos - 5 yrs' : 'Visits'}, inplace=True)
    clinics['Year'] = clinics['Year'].astype(int)



    # Get nearby schools for each clinic
    unique_school_geom = school_geom.drop_duplicates(subset=['geometry'])
    clinic_locs = np.array(list(clinics.geometry.apply(lambda x: (x.x, x.y))))
    school_locs = np.array(list(unique_school_geom.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(school_locs)
    dist, indices = btree.query(clinic_locs, k=100)

    # 0.5 to 1: 2/10 of visits -> 1/10 children
    # 1 to 2: 3/10 of visits -> 1/10 children
    # 2 to 3: 1/10 of visits -> 1/10 children
    # 3 to 4: 1/10 of visits -> 1/10 children
    # 4 to 5: 1/10 of visits -> 1/10 children
    # 5 to 6: 1/10 of visits -> 1/10 children
    ages = [1, 2, 3, 4, 5, 6]
    ratios = [1/10, 1/10, 1/10, 1/10, 1/10, 1/10]
    grade_to_name_th = {3 : '3rd', 4: '4th', 5: '5th', 6: '6th', 7: '7th', 8: '8th', 9: '9th'}

    test_years = {
        '2013-14' : 2014, '2014-15' : 2015, '2015-16' : 2016,
        '2016-17' : 2017, '2017-18' : 2018, '2018-19' : 2019
    }

    # Initialize number of students from RORCO at each school to 0
    school_geom['num_from_RORCO'] = 0

    def get_capacity(row):
        grade = int(row['Grade'].split()[-1])
        grade_th = grade_to_name_th[grade]
        return row[grade_th]

    school_geom['Capacity'] = school_geom.apply(lambda row: get_capacity(row), axis=1)

    # For each clinic, assign students to schools through the years
    for clinic_idx in range(len(clinics)):
        clinic_year = clinics['Year'][clinic_idx]
        clinic_visits = clinics['Visits'][clinic_idx]
        for age, ratio in zip(ages, ratios):
            # Compute number of children of this age
            total_num = int(clinic_visits * ratio)
            total_num = int(total_num / 6)
            # prevent from overcounting each child in the 6 years they appear
            # Convert age in year to grade in test_year
            # 4 year old in 2015 is 8 years old in 2019 which is 3rd grade
            for test_year in test_years:
                num = total_num
                grade = age + (test_years[test_year] - clinic_year) - 5
                # Check if grade is in school_geom
                if grade in grade_to_name_th:
                    grade_th = grade_to_name_th[grade]
                    for school_code_idx in indices[clinic_idx]:
                        school_code = unique_school_geom['School Code'].iloc[school_code_idx]
                        lookup = test_year + '_' + 'Grade 0' + str(grade) + '_' + str(school_code)                    
                        try:
                            school = school_geom.loc[lookup]
                        except KeyError:
                            continue
                        # Reserve 20% of capacity for non-RORCO students
                        capacity = school['Capacity'] * .8
                        num_from_RORCO = school['num_from_RORCO']
                        to_add = max(min(num, capacity-num_from_RORCO), 0)

                        col_num = school_geom.columns.get_loc('num_from_RORCO')
                        school_geom.iloc[school_geom.index.get_loc(lookup), col_num] += to_add


                        num -= to_add
                        if num == 0:
                            break

    school_geom['is_RORCO'] = (school_geom['num_from_RORCO'] / school_geom['Capacity'] > 0.75).astype(int)
    return school_geom

def add_proximity(school_geom):
    # Add nearby students and rural indicator
    # Convert geometry to meters
    school_geom = school_geom.to_crs('EPSG:3857')

    # Get nearby schools for each school
    numpy_array = np.array(list(school_geom.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(numpy_array)
    # Find all schools within 10 miles of each school (16093)
    nearby_indices = btree.query_ball_point(numpy_array, r=16093)

    # Saved school numbers for each school
    saved_schools = {}

    # Calculate number of students within 10 miles of each school
    nearby_students = []
    for school_index in range(len(school_geom)):
        current_school = school_geom['School Code'].iloc[school_index]
        if current_school not in saved_schools:
            num_students = 0
            for nearby_idx in nearby_indices[school_index]:
                num_students += school_geom['Capacity'].iloc[nearby_idx]
            saved_schools[current_school] = num_students
        nearby_students += [num_students]

    school_geom['nearby_students'] = nearby_students
    # Rural is defined as 50000 or fewer in area
    # Since 20% of Colorado population is children, that's 10,000 students
    school_geom['is_rural'] = (school_geom['nearby_students'] < 10000).astype(int)
    return school_geom

def plot_estimates(ns, run_estimates, included, title):

    colors = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000', 'red', 'black', 'blue', 'orange', 'purple', 'teal']
    for i, method_name in enumerate(run_estimates):
        mus = [np.mean(run_estimates[method_name][n]) for n in ns]
        sigmas = [np.std(run_estimates[method_name][n]) for n in ns]
        upper = [mu + sigma for mu, sigma in zip(mus, sigmas)]
        lower = [mu - sigma for mu, sigma in zip(mus, sigmas)]
        if method_name in included:
            plt.plot(ns, mus, label=method_name, linestyle=sf.linestyle_tuple[i][1], color=colors[i])
            plt.fill_between(ns, lower, upper, alpha=0.2, color=colors[i])
            print('Method Name:', method_name)
            print(mus[-1])
            print(sigmas[-1])

    plt.xlabel('Number of Observations')
    plt.ylabel('Estimate (CMAS Standard Deviations)')
    plt.title(title)
    plt.legend()
    plt.savefig(f'../images/{title}.pdf')
    plt.show()

if __name__ == '__main__':
    school_geom = load_school_data()

    clinics_geom = load_clinic_data()

    school_geom = assign_visits_to_schools(school_geom, clinics_geom)

    school_geom = add_proximity(school_geom)

    # Write to csv
    school_geom.to_csv('rorco_data.csv')