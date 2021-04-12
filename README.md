# Python_Pico_Reader
Low level PicoDst reader in native Python using uproot and awkward arrays. This needs so much work to be considered fully functional that it's laughable to call it a PicoDst reader, but it'll work for simple things.
Functionality is being added bit by bit, so check back (or, better yet, contribute!).

## Installation
Install with `pip install .`
If you are developing this package and would like to avoid running this after every change, instead use `pip install -e .` to create a link back to this directory. 

## Basic Usage

To create a pico reader instance:
```
import pico_reader
reader = pico_reader.PicoDST()
reader.import_data("path_to_picoDST")
```