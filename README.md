# Foobar

Foobar is a Python library for dealing with word pluralization.

## Installation


```bash
pip install -r requirements.txt
```

## Usage

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Functionality

Purpose:

This program takes in calcium imaging data of islets over time. The data has been quantified in ImageJ into time series data. The program takes this data into the porgram in wide format. The first Juytper Notebook "Graph_picker" reformats the data into a tidy format, normalizes the data, and acts as a tool to help select representitive graphs. The second Juytper Notebook anylizes the data including findings area under the first phase of the 11 mM glucose stimulus 1st phase. 

How to use:

Use the graph picker to normalize the data, give a preview of the data graphed, and genrate a file that guesses at the end of the 11 mM 1st phase. 