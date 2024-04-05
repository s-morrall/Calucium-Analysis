import numpy as np
import pandas as pd
import seaborn as sns
from math import log10, floor
import matplotlib.pyplot as plt  # Plotting library

import scipy  # Scientific and technical computing
from scipy.signal import find_peaks  # Function to find peaks in data
from scipy.signal import iirfilter
import matplotlib.ticker as ticker  # Import ticker module for custom tick formatting


sns.set_style("darkgrid")

class Format:
    def __init__(self, path = None):
        if (path == None):
            print("Error: No Path Defined, please provide a Path.")
        self.path = path

    
    def tidy_sheet(self, sheet, baseline, delta = False, average = False, errors = False):
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
        
        removed = 0 
        # For each cell in our list of cells, we shall find the starting absorbance, then normalize based on the time series data for that cell.
        for cell in cells:  

            
       
            # This normalizes based on the first numaric value in the time series data for that cell.
            i = 0
            if baseline == False:
                # Set the starting absorbance value to be later used for normalization.
                start_au = sheet[cell].iloc[0]

                while start_au == 0:
                    print(f"Normalization issue for {treatment} {cell} because first value is zero {sheet[cell].iloc[i]}")
                    i += 1
                    start_au = sheet[cell].iloc[i]
                
                normalized = sheet[cell] / start_au
            else:
            
                start_au = sheet[sheet["Time (sec)"] <= baseline][cell].mean().round(3)

                if start_au == 0:
                   
                    if errors == False:
                        continue

                    else:
                               
                        i = baseline - 5
                        while start_au == 0:
                            i += 5
                            start_au = sheet[sheet["Time (sec)"] <= (i)][cell].mean().round(3)
                            #DIV/0!

                        print(f"""Normalization issue for {treatment} {cell} because values in first {baseline} seconds are zero. 
                            Time frame would need to be expanded to {i} seconds. Values will be replaced with #DIV/0!""")  
            
                        normalized = "DIV/0!"
                       
                else:
                    normalized = sheet[cell] / start_au        

                
            # Create a dictionary containing all of the relevant data needed for a single cell.
            temp_df = pd.DataFrame({'Time (sec)': sheet["Time (sec)"], 
                        "[hr]:[min]:[sec]": sheet["[hr]:[min]:[sec]"],
                        'A.U.': sheet[cell], 
                        "Normalized A.U.":normalized,
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

    def tidy_file(self, delta = True, average = True, baseline = 15, errors = False):
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
            temp_df = self.tidy_sheet(sheet, baseline, delta, average, errors)

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
    
    
    def round_sig(self, x, sig=1, small_value=1.0e-9):
        """
        Rounds a number to a specified number of significant figures.

        Parameters:
        - x: The number to be rounded.
        - sig: The number of significant figures to round to (default is 1).
        - small_value: A threshold below which x is considered too small to be rounded normally (default is 1.0e-9).

        Returns:
        - The rounded number.
        """

        # Determine the number of digits to round to. This is done by:
        # 1. Calculating the logarithm base 10 of the absolute value of x,
        #    but replace x with small_value if x is smaller in magnitude than small_value to avoid log(0) error.
        # 2. Taking the floor of this value to get the largest integer less than or equal to the logarithm,
        #    which effectively determines the order of magnitude of x.
        # 3. Subtracting this value from the desired number of significant figures, adjusting for the index base.
        rounded = round(x, sig - int(floor(log10(max(abs(x), abs(small_value))))) - 1)

        return rounded
    
    
    def configure_filter_for_butterworth(self, mode: str = 'smooth') -> tuple[int, float]:
        """
        This function configures and returns parameters for a Butterworth low-pass filter.
        These parameters include the filter order and the normalized cutoff frequency,
        which are chosen based on the input mode that specifies the desired level of data smoothing.

        Parameters:
        - mode: A string that indicates the smoothing mode. It can be 'detailed', 'medium', or 'smooth',
                affecting the sharpness and the extent of smoothing applied by the filter.
                The default value is 'smooth'.

        Returns:
        - A tuple containing two elements:
        1. The order of the filter (int): Determines the steepness of the filter's response curve.
        2. The normalized cutoff frequency (float): Specifies the frequency at which
            the filter's power is reduced to half its passband value, relative to the Nyquist rate.

        Raises:
        - ValueError: If the mode provided does not match one of the predefined options.

        Example:
        Demonstrates how to use the returned values with the iirfilter function from scipy.signal
        to create a low-pass Butterworth filter.
        """

        # Dictionary mapping the mode to its corresponding filter parameters:
        # 'detailed', 'medium', and 'smooth' are supported modes.
        # Each mode is associated with a tuple containing the filter order and the cutoff frequency.
        mode_settings = {
            'detailed': (3, 0.3),  # Higher cutoff frequency, less smoothing
            'medium': (3, 0.2),    # Moderate cutoff frequency, moderate smoothing
            'smooth': (4, 0.02),   # Lower cutoff frequency, more smoothing
        }

        # Validate the input mode and raise an error if the mode is not supported.
        if mode not in mode_settings:
            raise ValueError(f"Unsupported mode '{mode}'. Supported modes are: {list(mode_settings.keys())}.")

        # Extract the filter order and cutoff frequency based on the selected mode.
        order, cutoff_freq = mode_settings[mode]

        # Return these parameters for use in configuring a Butterworth filter.
        return order, cutoff_freq


    def find_end_points(self, plot_df: pd.DataFrame, max_locations: list, min_locations: list, min_end_count: int = 3) -> list:
        """
        Identifies end points following specified starting points within a plot's data.
        It also guarantees a minimum number of end points by filling in additional points if necessary.
        
        Parameters:
        - plot_df: DataFrame containing the plot data. It must include a 'Time (sec)' column for time values.
        - max_locations: List of indices representing the starting points for finding end points. These are typically
                        the locations of maximum values from where the search for end points begins.
        - min_locations: List of indices representing potential end points. These are typically locations of
                        minimum values that can serve as ends following the max_locations.
        - min_end_count (optional): The minimum number of end points to find. If fewer are found, the list
                                    will be filled until this minimum is reached. Defaults to 3.
        
        Returns:
        - A list of times corresponding to the end points, ensuring at least a minimum number specified.
        The times are sorted in ascending order.
        """

        # Initialize an empty list to store end point times
        ends = []
        # Loop through each starting point index in max_locations
        for location in max_locations:
            # Search for the first subsequent end point from min_locations
            for l in min_locations:
                if l > location:
                    # Once an end point is found, add its time to the list and break the loop to move to the next start point
                    ends.append(plot_df.iloc[l]["Time (sec)"])
                    break

        # If not enough end points were found, add more from min_locations or fill with NaN
        i = 0  # Initialize counter for accessing min_locations
        while len(ends) < min_end_count:
            if len(min_locations) > i:
                # Extract time for the current min_location index
                time = plot_df.iloc[min_locations[i]]["Time (sec)"]
                # Ensure no duplicate times are added
                if time not in ends:
                    ends.append(time)
                i += 1  # Move to the next index in min_locations
            else:
                # If there are not enough min_locations, fill the remaining slots with NaN
                ends.append(np.nan)
        # Sort the end times in ascending order
        ends.sort()

        return ends

    def apply_butterworth_filter(self, plot_df: pd.DataFrame, column_name: str = "Normalized A.U.", mode: str = 'smooth') -> pd.DataFrame:
        """
        Enhances the input DataFrame by smoothing data in a specified column using a Butterworth low-pass filter.
        The filtered data is added as a new column, enabling comparison between original and smoothed data.

        Parameters:
        - plot_df: DataFrame containing the data to be filtered. It must include a column with the name specified
                by the 'column_name' parameter, containing numerical data to smooth.
        - column_name: The name of the column to apply the filter on. This column should contain the data you wish
                    to smooth, typically some form of measurement data.
        - mode: Determines the filter's smoothing characteristics. Supported modes are 'detailed', 'medium', and 'smooth',
                affecting the filter's order and cutoff frequency.

        Returns:
        - The original DataFrame with an additional column containing the smoothed data. This new column is named
        'Smoothed [Original Column Name]'.

        The function configures a Butterworth filter based on the desired smoothing mode, applies this filter to
        the specified column of data, and inserts the smoothed data as a new column in the DataFrame.
        """

        # Configure the Butterworth filter parameters based on the desired smoothing mode.
        order, cutoff_freq = Analyze.configure_filter_for_butterworth(self, mode)

        # Design the Butterworth filter using the configured parameters.
        b, a = scipy.signal.iirfilter(N=order, Wn=cutoff_freq, btype='low', ftype='butter')

        # Apply the filter to the specified column of data.
        # The filtfilt function is used for zero-phase filtering, which avoids phase shift in the filtered data.
        smoothed_AU = scipy.signal.filtfilt(b, a, plot_df[column_name])

        # Insert the smoothed data as a new column in the DataFrame.
        # The column is inserted in the third position (index 2), with the column name indicating it contains
        # smoothed data. If a column with this name already exists, it will be overwritten.
        plot_df.insert(2, "Smoothed " + column_name, smoothed_AU, True)

        # Return the DataFrame with the added column of smoothed data.
        return plot_df


    
class Graph:

    def __init__(self):
        return 
    
    def mark_locations(self, ax: plt.Axes, plot_df: pd.DataFrame, locations: list, mark: str, color: str):
        """
        Marks specified locations on a plot with either a vertical line or a dot, depending on the 'mark' parameter.
        
        Parameters:
        - ax (plt.Axes): The matplotlib Axes object where the markings are to be made. This is the plot area.
        - plot_df (pd.DataFrame): The DataFrame containing the data that has been plotted. It must have
                                a column labeled 'Time (sec)' which indicates the time points of the data.
        - locations (list): A list of indices pointing to rows in plot_df. These indices specify the data points
                            that should be marked on the plot.
        - mark (str): Specifies the style of marking to use. This can either be 'dot' for marking with dots, or
                    'line' for marking with vertical lines.
        - color (str): The color of the markers or lines. This can be any color format recognized by matplotlib.
        
        Raises:
        - ValueError: This exception is raised if the 'mark' parameter is neither 'dot' nor 'line', ensuring that
                    only valid marking styles are used.
        """

        # Iterate through each location index provided in the 'locations' list.
        for location in locations:
            # Extract the time value from the 'Time (sec)' column of the DataFrame for the current location.
            time = plot_df.iloc[location]["Time (sec)"]
            
            # Check if the marking style is 'dot'.
            if mark == 'dot':
                # Extract the corresponding y-value for the plot; here it's assumed to be "Smoothed Normalized A.U."
                # You might need to adjust this depending on the data you're plotting.
                au = plot_df.iloc[location]["Smoothed Normalized A.U."]
                # Plot a dot at the specified location using the provided color and a fixed marker size.
                ax.plot(time, au, marker='o', color=color, markersize=5)
                
            # If the marking style is 'line', draw a vertical line at the specified time.
            elif mark == 'line':
                ax.axvline(x=time, color=color, linestyle='--')
                
            # If the mark parameter is neither 'dot' nor 'line', raise a ValueError.
            else:
                raise ValueError("mark parameter must be 'dot' or 'line'.")

    def mark_plot_and_update_ends(self, plot_df: pd.DataFrame, min_locations: list, max_locations: list, 
                                temp_md: pd.DataFrame, cell: str, ax: plt.Axes, end_count: int = 3) -> pd.DataFrame:
        
        """
        This function enhances a plot by marking specified points and updates a metadata DataFrame with
        end points calculated based on the provided max and min locations in the data.

        Parameters:
        - plot_df: The DataFrame containing the data to be plotted. It is expected to include a 'Time (sec)' column.
        - min_locations: Indices of the minimum points in plot_df that will be marked on the plot.
        - max_locations: Indices of the maximum points in plot_df that will be highlighted.
        - temp_md: The DataFrame to update with end point options. This could be a metadata table that tracks
                analysis outcomes for different cells or conditions.
        - cell: A string identifier for the specific cell or condition being analyzed, used to update temp_md.
        - ax: The Matplotlib Axes object where the data points are to be plotted.
        - end_count: The number of end points to find and mark, with a default value of 3.

        Returns:
        - An updated copy of temp_md DataFrame with new columns for each end option found, plus additional
        metadata related to the analysis.

        The function first uses 'mark_locations' to visually mark the min and max locations on the plot.
        Then, it calculates end points following the max locations using 'find_end_points'. These end points
        are marked on the plot, and the temp_md DataFrame is updated with this information for the specified cell.
        """


        # Mark the minimum locations with lines of a specified color (light grey here) on the plot
        self.mark_locations(ax, plot_df, min_locations, 'line', 'lightgrey')

  
        # Similarly, mark the maximum locations with dots of a specified color (red here)
        self.mark_locations(ax, plot_df, max_locations, 'dot', 'red')

        # Calculate end points based on the provided max and min locations
        ends = Analyze.find_end_points(self, plot_df, max_locations, min_locations, end_count)

        # Make a copy of the input DataFrame to avoid altering the original data
        cell_md_cop = temp_md.copy()
        # Update the copy with the identifier for the current analysis
        cell_md_cop["Example Cells"] = cell

        # Mark each end point on the plot and update the DataFrame with these times
        for i, time in enumerate(ends):
            ax.axvline(x=time, color='black')  # Mark the end point with a vertical line
            cell_md_cop[f"End Option {i + 1}"] = ends[i]  # Update the DataFrame with the end option

        # Additional metadata updates
        cell_md_cop["11 mM End"] = ""  # Placeholder for further data
        cell_md_cop["Represent"] = "TRUE"  # Mark the cell's data as representative
        
        # Return the updated DataFrame
        return cell_md_cop
    
    def customize_axis(self, ax, plot_df):
        """
        This function customizes the axes of a Matplotlib plot to improve its readability and aesthetics.
        It specifically focuses on adjusting the tick locators and label sizes for both the x-axis and y-axis.

        Parameters:
        - ax: The Matplotlib Axes object that will be customized. This object represents the axes of the plot
            on which the data is drawn.
        - plot_df: A pandas DataFrame that contains the data series to be plotted. This DataFrame must include
                a column named "Normalized A.U." from which the function calculates spacing for y-axis ticks.

        The function modifies the major and minor tick locators for both axes and adjusts the tick label font sizes.
        It enables minor ticks to provide a finer scale on the axes, improving the plot's detail level.
        """

        # For the x-axis, set the interval of major ticks to 100 units.
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
        
        # For the y-axis, determine the spacing of major ticks based on the maximum value in the "Normalized A.U." column.
        # The maximum value is divided by 20 and rounded to two decimal places to establish a reasonable interval.
        yspacing = round((plot_df["Normalized A.U."].max() / 20), 2)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(yspacing))
        
        # Adjust the font size of tick labels on both axes to size 10 for improved readability.
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        # Enable minor ticks on both axes to provide additional scale granularity.
        ax.minorticks_on()
        # Configure the x-axis to display one minor tick between the major ticks.
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        # Customize the appearance of minor ticks on the x-axis.
        ax.tick_params(axis='x', which='minor', bottom=True, top=False, length=5)

        # Perform a similar configuration for the y-axis' minor ticks.
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        # The y-axis minor tick parameters are commented out as they might duplicate the settings for major ticks,
        # but typically, you might adjust 'which' to 'minor' for specific styling.


    def configure_plot_ticks(self, ax, plot_df, x_spacing_ratio, y_spacing_ratio, x_column_name="Time (sec)"):
        """
        Adjusts tick intervals and styles on a Matplotlib plot based on data-driven calculations.

        Parameters:
        - ax: The Matplotlib Axes object to be configured. This is where the data will be plotted.
        - plot_df: A pandas DataFrame containing the data that influences the tick configuration.
                The DataFrame should contain at least one column specified by x_column_name and
                a "Normalized A.U." column for y-axis calculations.
        - x_spacing_ratio: A numeric value that determines how the spacing between major ticks on the
                        x-axis is calculated. The plot's x-range is divided by this ratio.
        - y_spacing_ratio: Similar to x_spacing_ratio, but used to calculate spacing between major ticks
                        on the y-axis.
        - x_column_name: Specifies which column from plot_df should be used for x-axis calculations. This
                        allows for flexibility in adapting to different data structures.

        The function dynamically adjusts major tick intervals for both axes to suit the dataset's range,
        ensuring the plot is neatly organized and easily interpretable. Minor ticks are also added for
        additional detail.
        """

        # Calculate the spacing for major ticks on the x-axis based on the specified ratio and data range.
        if x_column_name == "Time (sec)" and x_spacing_ratio != False:
            # Use the round_sig function to round the calculated spacing to a significant figure.
            xspacing = int(Analyze.round_sig(self, (plot_df[x_column_name].max() - plot_df[x_column_name].min()) / x_spacing_ratio))
        else:
            # If x_spacing_ratio is False or another column name is specified without logic, default to a fixed value.
            xspacing = 100  # This default value can be adjusted as needed.

        # Apply the calculated x-spacing to the plot's x-axis major tick locator.
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xspacing))

        # Calculate and set the spacing for major ticks on the y-axis, similar to the x-axis.
        if y_spacing_ratio != False:
            yspacing = plot_df["Normalized A.U."].max() / y_spacing_ratio
            yspacing = Analyze.round_sig(self, yspacing, sig=1)  # Round to 1 significant figure for clarity.
            ax.yaxis.set_major_locator(ticker.MultipleLocator(yspacing))
        
        # Set the font size for both axes' tick labels to improve readability.
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        # Enable minor ticks to provide more detail between major ticks.
        ax.minorticks_on()
        # Configure minor tick locators for both axes. This sets one minor tick between each pair of major ticks.
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        # Specify the appearance of minor ticks on the x-axis.
        ax.tick_params(axis='x', which='minor', bottom=True, top=False, length=5)
        # Additional customization for y-axis minor ticks could be added similarly if necessary.




                    
                # def fourier_smooth(self, time)

                        

                    
