import numpy as np
import pandas as pd
import seaborn as sns


sns.set_style("darkgrid")

class Format:
    def __init__(self, path = None):
        if (path == None):
            print("Error: No Path Defined, please provide a Path.")
        self.path = path

    
    def tidy_sheet(self, sheet, baseline, delta = False, average = False):
        """
        This is a function to take in one sheet of data as formated and make it tidy.
        :param boolean delta: A boolean to signift is using Delta. 
        :param boolean average: A boolean to signify if using average. 
        :return dataframe: Returns a pandas dataframe. 
        """
        
        # Drops all columns that contain Null values. 
        sheet.dropna(how= "all", axis=1, inplace=True)
        # Drops all rows that contain Null values. 
        sheet.dropna(how= "all", axis=0, inplace=True)
        
        # Extract the run conditions of the sample from {machine}. (Note, for our unit this is the top right cell)
        treatment = sheet.iloc[0,0]
        
        # Rename the columns using the second row (index 1) of the sheet which should contain the cell names.
        sheet.rename(columns=sheet.iloc[1], inplace = True)
        
        # Remove a mostly blank row that will be left after massaging the data, as well as removing the column name row.
        sheet.drop(sheet.index[0:2], inplace = True)
        sheet.reset_index(drop=True, inplace=True)

        # A Bulion array that finds the columns that contain "Cell" within them. 
        mask = sheet.columns.str.contains("Cell", na=False)
        
        # Takes the mask, and makes an array of these column names.
        cells = sheet.columns[mask]
        
        # Cast cells to a list. 
        cells = list(cells)

        # If we have enabled the Average boolean, then we will now find the average of each Cell.  
        if average == True:
            sheet["Cell Avg"] = sheet.loc[:, mask].mean(axis=1)
            cells.append("Cell Avg")

        # If the sheet contains ^AVG$, then append AVG for easier manipulation. 
        if sheet.columns.str.contains('^AVG$', regex=True, na=False).any() == True:
            cells.append("AVG")

        temp_dfs = []
        
        # For each cell in our list of cells, we shall find the starting absorbance, then normalize based on the time series data for that cell.
        for cell in cells:  

            # Set the starting absorbance value to be later used for normalization.
            start_au = sheet[cell].iloc[0]
       
            
            # This normalizes based on the first numaric value in the time series data for that cell.
            i = 0
            if baseline == False:
                while start_au == 0:
                    print(f"Normalization issue for {treatment} {cell} because first value is zero {sheet[cell].iloc[i]}")
                    i += 1
                    start_au = sheet[cell].iloc[i]
       
            else:
                start_au = sheet[sheet["Time (sec)"] <= baseline][cell].mean().round(3)
         
                if start_au == 0:
                    i = baseline - 5
                    while start_au == 0:
                        i += 5
                        start_au = sheet[sheet["Time (sec)"] <= (i)][cell].mean().round(3)
                        #start_au = np.mean(sheet[sheet["Time (sec)"] <= i][cell])
                  
                    print(f"""Normalization issue for {treatment} {cell} because values in first {baseline} seconds are zero. 
                          Time frame was expanded to {i} seconds.""")        

            # Create a dictionary containing all of the relevant data needed for a single cell.
            temp_df = pd.DataFrame({'Time (sec)': sheet["Time (sec)"], 
                        "[hr]:[min]:[sec]": sheet["[hr]:[min]:[sec]"],
                        'A.U.': sheet[cell], 
                        "Normalized A.U.": sheet[cell] / start_au,
                        'Cell #': cell, 
                        'Treatment': treatment})
            
            # If Delta has been enalb,ed this adds the delta to the normalized A.U. 
            if delta == True:
                temp_df = Analyze.add_delta(self, temp_df, colm = "Normalized A.U.")

            # Account for differnet times in each cell. 
            temp_df.dropna(subset=['A.U.'], inplace=True)


            # Append our new dataframe to our set of temp dataframes. 
            temp_dfs.append(temp_df)

    
        # Create a master list of dataframes of all of our subsets of temp dataframe. 
        df = pd.concat(temp_dfs, ignore_index=True)

        return df

    def tidy_file(self, delta = True, average = True, baseline = 15):
        """
        This is a function to take in an entire excelt file and tidy it. 
        :param boolean delta: A boolean to signift is using Delta. 
        :param boolean average: A boolean to signify if using average. 
        :return dataframe: Returns a pandas dataframe. 
        """

        #creats a dictionary of all of the sheets in the file in the format of sheet name: sheet
        sheet_dict = pd.read_excel(self.path, sheet_name=None)

        temp_dfs = []
        #loops through all of the sheets in the excel file
        for sheet_name, sheet in sheet_dict.items():


            #tidiee one sheet for each loop through
            temp_df = self.tidy_sheet(sheet, baseline, delta, average)

            temp_dfs.append(temp_df)

        #combines all of the temp dfs to one large df
        df = pd.concat(temp_dfs, ignore_index=True)
        return df
        
class Analyze:

    def __init__(self):
        return 
    
    def add_delta(self, df, colm = "Normalized A.U."):
        """adds a column to the df that shows the change between each row"""
        deltas = [0]
        for i in range(1, len(df["Normalized A.U."])):
            current = list(df["Normalized A.U."])[i]
            last = list(df["Normalized A.U."])[i-1]

            deltas.append(current - last)

        df["delta"] = deltas

        return df
    


    def AUC(self, md_df, data, start = "11 mM Stimulus", end = "11 mM End", style="Trap", normal = False):
        # Initialize a list to store AUC values for each row in md_df but only works rn with only one typle of cell in each catagory 
        AUCs = []

        # Iterate over each row in md_df
        for i, row in md_df.iterrows():
            # Filter 'data' dataframe to include only rows where 'Treatment' matches the current row in md_df
            min_df = data[data["Treatment"] == row["Treatment"]]

            # Further filter 'min_df' to include only rows where 'Time (sec)' is between 'Start' and 'End' of the current row in md_df
            min_df = min_df[min_df["Time (sec)"].between(row[start], row[end])]
                    # Check if min_df is empty

            if min_df.empty:
             
                AUCs.append(np.nan)  # Append NaN or some other placeholder
                continue
            # Extract 'Time (sec)' column as x-values
            x = min_df["Time (sec)"].values

            if normal == False:
                # Calculate y-values by subtracting the baseline (first value) from 'Normalized A.U.'
                y = min_df['Normalized A.U.'] - min_df['Normalized A.U.'].iloc[0]
            if normal == True:
                y = min_df['Normalized A.U.']/min_df['Normalized A.U.'].iloc[0]



            # Choose the method to calculate AUC based on the 'type' parameter
            if style.lower() == "trap":
                # Calculate the AUC using the trapezoidal rule
                AUC_value = np.trapz(y, x)
            elif style.lower() == "rect":
                # Calculate the AUC using the rectangular method
                y = y.values
                AUC_value = 0
                for i in range(1, len(y)):
                    # Calculate the width of each rectangle (delta_x)
                    delta_x = (x[i] - x[i-1])
                    # Height of the rectangle (delta_y)
                    delta_y = y[i-1]
                    # Incrementally add the area of each rectangle to 'AUC_value'
                    AUC_value += delta_x * delta_y

            # Append the calculated AUC to the AUCs list
            AUCs.append(AUC_value)
            
        if style == "rect":
            md_df["AUC (rect)"] = AUCs
            return md_df

        # Add a new column 'AUC' to md_df containing the calculated AUCs
        md_df["AUC (trap)"] = AUCs
        # Return the modified md_df with AUC values
        return md_df

    
   # def fourier_smooth(self, time)

        

      
