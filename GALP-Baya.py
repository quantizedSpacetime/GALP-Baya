# -*- coding: utf-8 -*-

import sys
import os.path
import subprocess
import re

import numpy as np

from astropy.io import fits

import cobaya
from cobaya.run import run
from cobaya.likelihood import Likelihood
from cobaya.theory import Theory
from cobaya.yaml import yaml_load_file

# used for profiling the code's efficiency
import cProfile

"""

૮₍ ´• ˕ •` ₎ა

⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⡀⠀⠖⠲⠤⣄⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢀⣤⣄⣀⠀⠠⠚⠋⠀⠁⠈⠙⣢⡤⠖⡄⠀⠈⠉⠛⢦⣀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢠⣿⣿⣿⡾⢷⢦⡀⠀⠀⠀⣠⠾⠉⠀⠀⣿⠀⠀⠀⢀⣠⣶⣥⡄⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣰⡿⠿⠿⣿⣿⣿⣀⠉⢶⡀⣸⣿⡀⠀⠀⠀⣹⡆⠀⢸⣿⣿⣿⣿⣿⣷⡀⠀⠀⠀⠀
⠀⠀⠀⠀⢻⣿⣟⣷⣾⣿⣿⣿⣷⠞⠀⣹⣿⠃⠀⠀⠀⠈⣧⠀⠉⣿⣿⣿⣿⣿⣻⢿⣄⣀⠀⠀
⠀⠀⠀⠀⠀⠻⣿⣿⣿⣿⣿⣿⣿⣆⣠⡟⠂⠀⠀⠀⠀⠀⢸⠀⣤⢿⣿⣿⣿⣿⣿⣿⡿⠛⠁⠀
⠀⠀⠀⠀⡠⠀⣾⣿⣿⣿⣿⣿⣿⣿⣿⠇⠀⠀⠀⠀⠀⠀⠘⣾⣿⣿⢿⣿⣿⣿⣿⣧⠤⠀⠀⠀
⠀⠀⠀⡼⠁⣸⣿⣿⣿⡟⠋⠁⣽⣿⠟⠀⠀⠀⠀⠀⠀⠀⠀⠛⣿⡿⠀⠈⠙⣻⡿⡦⠀⢰⠀⠀
⠀⠀⣼⠁⢀⣽⠿⠛⠉⠀⠀⠰⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠀⠀⠀⠀⠠⠤⡀⠀⣇⠀
⠀⢰⠇⠀⣺⠿⠋⠀⠀⠀⠀⠀⠀⠀⠈⠦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠼⡄
⠀⡾⠀⠀⠀⣁⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠤⢦⣄⢀⠠⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇
⠘⡇⠀⠀⠀⠳⠄⠀⠀⠀⠀⠀⠀⠀⠀⠸⢶⣦⢩⢠⣷⠾⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇
⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣙⣾⣏⣽⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇
⠀⣏⠀⠀⠀⠀⠐⠲⢀⡀⠀⠀⠀⠀⠀⠀⠀⠘⠻⠿⠍⠀⠀⠀⠀⠀⠀⠀⢀⠤⠐⠀⠀⠀⣼⠀
⠀⢸⡀⠸⡆⠀⠀⠉⠀⠿⢏⡷⡆⠀⠀⠀⠀⠀⠀⠸⠇⠀⠀⠀⠀⠀⣀⠰⢉⡁⡀⠀⠀⢀⡏⠀
⠀⠈⣷⠀⠳⠀⠠⠐⡁⡖⢠⡉⡙⠶⢀⡀⢤⠀⠀⠸⢦⠀⠀⠀⠀⣀⣠⣤⠄⠉⠳⠄⠀⢼⠁⠀
⠀⠀⠘⢧⠀⠇⢀⠁⡒⢌⡩⢳⣧⣣⢿⣷⢸⡆⠘⣆⠀⢳⡀⢧⣿⣿⣿⣆⠱⣀⠀⠀⠀⣼⠀⠀
⠀⠀⠀⠀⣰⣦⡐⣦⣍⢲⠥⣿⣻⣿⣿⣿⣿⣿⣲⣯⣤⣿⣶⣿⣿⣿⣿⣿⣆⡿⠀⡀⢠⡟⠀⠀
⠀⠠⠤⠤⠽⠭⢤⢶⣿⣮⣿⣏⣿⣹⡟⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠛⣷⡿⣝⣶⡷⠞⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⢉⣉⠉⠉⠀⠀⠀⠀⠀⠀

"""

# ==================================
# GLOBAL PARAMETER DEFINITIONS
# ==================================

# loggerLevel 0 = Errors only, 1 = Errors and warnings only, 2 = Full, 3 = Debug
loggerLevel = 0

galdef_filepath = 'galdef_57_fullrun'
galdef_name = 'fullrun'
src_filepath = 'source_class_fullrun.txt'

# needed to modify eval_iso_cs.dat for Xsec norm
galtools_share_path = '/mnt/c/Users/zacha/Documents/Cosmic_Ray/GALPROPv57Files/galprop_v57_release_r1/lib/build/galtoolslib-1.1.1006/share'

galprop_path = '/mnt/c/Users/zacha/Documents/Cosmic_Ray/GALPROPv57Files/galprop_v57_release_r1/build/source/galprop'
results_path = './RESULTS'
FITS_path = '../../../FITS'
galdef_path = '.'

dataUsine_filepath = 'Combined_v4.usine'

cobaya_yaml_filepath = 'totalYaml.yaml'
interface_yaml_filepath = 'totalInterface.yaml'

# ==================================
# DO NOT MODIFY
# ==================================

nuclei = {}

usineToZ = {'H' : [1],
            'HE' : [2],
            'LI' : [3],
            'BE' : [4],
            'B' : [5],
            'C' : [6],
            'N' : [7],
            'O' : [8],
            'F' : [9],
            'NE': [10],
            'NA' : [11],
            'MG' : [12],
            'AL' : [13],
            'SI' : [14],
            'P' : [15],
            'S' : [16],
            'CL' : [17],
            'AR' : [18],
            'K' : [19],
            'CA' : [20],
            'SC' : [21],
            'TI' : [22],
            'V' : [23],
            'CR' : [24],
            'MN' : [25],
            'FE' : [26],
            'CO' : [27],
            'NI' : [28],
            'SUBFE' : [21,22,23,24,25]}

# ==================================
# Utility FUNCTIONS
# ==================================

# Utility function used to record messages to the console
# lvl <= 0 for errors, =1 for warnings, 2 for general info,
# 3 for full debugging
# lvl < 0 used to force exit on error
#
# Parameters: message (str) message to be printed to the console
#    lvl (int) indicates "severity" of message, see above
#    errcode (int, opt) error code to return for lvl < 0
# Returns: None
def logMessage(message, lvl, errcode=1):

   messageLevels = ['ERROR: ', 'Warning: ', 'Info: ', 'Debug: ']

   ind = lvl

   if ind < 0:
      ind = 0
   if ind >= len(messageLevels):
      ind = len(messageLevels) - 1

   if ind <= loggerLevel:
      print(messageLevels[ind] + message)
   if lvl < 0:
      print(f'Program exiting unsuccessfully, errcode={errcode}')
      sys.exit(errcode)

# Check if a unit is recognized by the program
# currently rigidity, kinetic energy per nucleon, kinetic energy, and total
# energy are recognized
#
# Parameters: unit (str) unit to be checked for validity
# Returns: True or False
def isUnitValid(unit):
   validUnits = ['R', 'EKN', 'EK', 'ETOT']
   return (unit in validUnits)

# Generate a dictionary format copy of a usine file
# used in this code to translate experimental data into a python-readable format
# Positive and negative error bars are allowed to be different, and data can be
# represented as an upper limit with the negative error bar treated as np.inf
# In cases when both statistical and systematic errors are provided, these
# are added in quadrature to produce the total error.
#
# Parameters: usineFilename (str) path to usine file to be read
# Returns: A dict with 2-tuples of the quantity name and experiment name as keys
#    The value for each key is a dict with 'Ebins', 'Value', 'Err-', 'Err+',
#    'upLim', 'phi', and 'dist' as keys. Phi and dist are individual numbers
#    while the others are numpy arrays
def generateUsineDict(usineFilename):
   logMessage(f'Begin reading usine file {usineFilename}', 2)
   usineDict = {}

   if not os.path.isfile(usineFilename):
      logMessage(f'Usine file {usineFilename} could not be found', -1)
      return usineDict

   usineRaw = np.genfromtxt(usineFilename, delimiter=None, \
                            dtype=str)

   for l in usineRaw:
      if not isUnitValid(l[2]):
         logMessage('Unknown unit type detected in usine dictionary generation', 1)
         logMessage(f'Usine filename {usineFilename}, unit {l[2]}, Key {(l[0].upper(), l[1])}', 1)
         logMessage('Usine dictionary generation will proceed but correctness of results is not guaranteed.', 1)
      newKey = (l[0].upper(), l[1])
      if not newKey in usineDict.keys():
         usineDict[newKey] = {'Unit' : l[2],
                            'Ebins' : [],
                            'Value' : [],
                            'Err-' : [],
                            'Err+' : [],
                            'upLim' : [],
                            'phi' : float(l[12]),
                            'dist' : float(l[13])}

      usineDict[newKey]['Ebins'].append(float(l[3]))
      usineDict[newKey]['Value'].append(float(l[6]))
      # note: statistical and systematic errors are added in quadrature for simplicity
      usineDict[newKey]['Err-'].append(np.sqrt(float(l[7])**2 + float(l[9])**2))
      usineDict[newKey]['Err+'].append(np.sqrt(float(l[8])**2 + float(l[10])**2))
      usineDict[newKey]['upLim'].append(float(l[15]))

   for e in usineDict.keys():
      for v in ['Ebins', 'Value', 'Err-', 'Err+', 'upLim']:
         usineDict[e][v] = np.asarray(usineDict[e][v])

   logMessage(f'Keys from usine file {usineFilename}: ', 3)
   logMessage(str(usineDict.keys()), 3)
   logMessage('Usine file successfully read in', 2)
   return usineDict

# Generate a list of Dataset objects from a usine file with experimental data
# For each key in a generated usine dict (see func. generateUsineDict), we
# create one Dataset object. See Dataset for more info
#
# Note: By default, the Dataset's modulationInfo member is set to 'IS' if
# the 'dist' value is >= 120 AU, otherwise it's set to 'FF'. The phi column of
# the USINE file is not used currently, although it is read in from the USINE
#
# Parameters: usineFilename (str) path to usine file to be read
# Returns: A list of Dataset objects, one for each experiment name / quantity name
#    pair in the original file
def listDatasetsUsine(usineFilename):
   logMessage(f'Generating list of datasets from usine file {usineFilename}', 2)
   udict = generateUsineDict(usineFilename)
   datasets = []
   for k in udict.keys():
      datasets.append(Dataset(k, udict[k], ('IS' if udict[k]['dist'] >= 120 else 'FF')))
   logMessage('Datasets successfully generated from usine file', 2)
   return datasets

# inserts a value of a parameter into a line in a galdef or source class file
# eg. inserting 2e3 into position 1 of spectral_pars = 1.8 3e3 2.4 produces
# spectral_pars = 1.8 2e3 2.4
#
# Parameters: string (str) line in the file that we want to modify
#    value: new value to add in
#    position: if string contains an array of values, which one are we changing
#       (zero-indexed)
# Returns: str with new values added in
def insertValue(string, value, position):
   string_nocomment = string.split('#')[0]
   # by splitting off the comments we can attach them back on later
   string_comment = string.split('#')[1:]
   # isolate the values on the right hand side of the equals sign
   string_rhs = string_nocomment.split('=')[1]
   # and split on whitespace to handle arrays of values
   string_split = re.split(' +', string_rhs)
   # handle edge case behavior of the .split function
   if string_split[0] == '':
      string_split = string_split[1:]

   logMessage(f'Replacing {string_split[position]} with {str(value)}', 3)
   if '\n' in string_split[position]:
      string_split[position] = str(value) + '\n'
   else:
      string_split[position] = str(value)

   # here we re-attach everything back together
   result = ''
   for s in string_split:
      result = result + ' ' + s
   for s in string_comment:
      result = result + '#' + s

   return result

# inserts a value of a parameter into a line in the eval_iso_cs.dat file
# contained in GALPROP. This is useful for re-normalizing production cross sections.
#
# Parameters: string (str) line in the file that we want to modify
#    value: new value to add in
#    position: if string contains an array of values, which one are we changing
#       (zero-indexed)
# Returns: str with new values added in
def insertXsecValue(string, value, position):
   string_nocomment = string.split('!')[0]
   # by splitting off the comments we can attach them back on later
   string_comment = string.split('!')[1:]
   # split on whitespace to handle arrays of values
   string_split = re.split(' +', string_nocomment)
   # handle edge case behavior of the .split function
   if string_split[0] == '':
      string_split = string_split[1:]

   logMessage(f'Replacing {string_split[position]} with {str(value)}', 3)
   if '\n' in string_split[position]:
      string_split[position] = str(value) + '\n'
   else:
      string_split[position] = str(value)

   result = '    '
   for s in string_split:
      result = result + '  ' + s
   for s in string_comment:
      result = result + '!' + s

   return result

# set all the parameters in the dictionary pvs
# used in theory and likelihood calculations to set parameters prior to running
# GALPROP / modulation codes
# For each parameter in pvs, we check for dependents first and set those before
# finally setting the parameter itself
#
# Parameters: pvs (dict) keys are parameter names, values are the values
#    (these dictionaries are provided by Cobaya samplers)
# Returns: None
def setParameters(pvs):

   alreadySet = []

   for p in pvs.keys():
      if p in interfaceInfo.keys():
         if 'dependents' in interfaceInfo[p].keys():
            for q in interfaceInfo[p]['dependents']:
               if q not in alreadySet:
                  setParameter(q, eval(interfaceInfo[q]['lambda']))
                  alreadySet.append(q)
         # we always set parameters regardless
         setParameter(p, pvs[p])
         alreadySet.append(p)

def setParameter(info_name, value):

   paramType = interfaceInfo[info_name]['type']

   galpName = None
   if 'galpName' in interfaceInfo[info_name].keys():
      galpName = interfaceInfo[info_name]['galpName']

   position = 0
   if 'position' in interfaceInfo[info_name].keys():
      position = interfaceInfo[info_name]['position']

   if paramType == 'galdef':
      logMessage(f'Inserting {paramType} parameter {info_name} at value {value}', 3)
      with open(galdef_filepath, 'r') as f:
         lines = f.readlines()

      with open(galdef_filepath, 'w') as f:
         string1 = galpName + ' '
         string2 = galpName + '='
         for l in lines:
            if l.startswith(string1) or l.startswith(string2):
               f.write(f'{string1} = {insertValue(l, value, position)}')
            else:
               f.write(l)
   elif paramType == 'source':
      logMessage(f'Inserting {paramType} parameter {info_name} at value {value}', 3)

      src_file = src_filepath
      # allows for multiple source class files to be used, by default use config file
      if 'src_file' in interfaceInfo[info_name].keys():
         src_file = interfaceInfo[info_name]['src_file']

      with open(src_file, 'r') as f:
         lines = f.readlines()

      with open(src_file, 'w') as f:
         string1 = galpName + ' '
         string2 = galpName + '='
         for l in lines:
            if l.startswith(string1) or l.startswith(string2):
               f.write(f'{string1} = {insertValue(l, value, position)}')
            else:
               f.write(l)
   elif paramType == 'modulation_FF':
      logMessage(f'Inserting {paramType} parameter {info_name} at value {value}', 3)
      # empty for now
   elif paramType == 'Xsec_prod_norm':
      logMessage(f'Inserting {paramType} parameter {info_name} at value {value}', 3)

      progenitor = interfaceInfo[info_name]['progenitor']
      product = interfaceInfo[info_name]['product']

      eval_iso_path = f'{galtools_share_path}/eval_iso_cs.dat'

      logMessage(f'Got progenitor {progenitor} and product {product}', 3)

      with open(eval_iso_path, 'r') as f:
         lines = f.readlines()

      with open(eval_iso_path, 'w') as f:
         for line in lines:
            l = re.split(r' +', line)
            if len(l) > 4:
               if l[1] == progenitor and l[2] == product:
                  new_value = value * float(l[4])
                  logMessage(f'Changing {l[4]} into {new_value}', 3)
                  logMessage(f'Original line {line}', 3)
                  f.write(f'{insertXsecValue(line, new_value, 3)}')
               else:
                  f.write(line)
            else:
               f.write(line)
   else:
      logMessage('Parameter {info_name} not of recognizable type: {paramType}', 1)
      logMessage('We will continue but this may be an issue', 1)

# converts the name of a quantity in USINE format to a list of isotope names
# that can be extracted from a FITS file produced by GALPROP v57.
# eg. usineToFitsName('B') = ['Boron_10', 'Boron_11']
# Secondary and tertiary particles are included as necessary,
# eg. usineToFitsName('1H') = ['Hydrogen_1', 'secondary_protons']
# Additionally, the special names ALLPARTICLE and SUBFE are accepted. SUBFE
# refers to all nuclei with Z=21-25.
#
# Parameters: usineName (str) name of quantity in USINE format
# Returns: List of isotope names that can be found in GALPROP FITS file
def usineToFitsName(usineName):
   result = []

   if usineName.upper() == 'ALLPARTICLE':
      for k in nuclei.keys():
         result.append(k)
   elif usineName.upper() == 'ELECTRONS':
      for k in nuclei.keys():
         if 'electron' in k:
            result.append(k)
   elif usineName.upper() == 'POSITRONS':
      for k in nuclei.keys():
         if 'positron' in k:
            result.append(k)
   elif usineName.upper() == '1H-BAR':
      for k in nuclei.keys():
         if 'antiproton' in k:
            result.append(k)
   else:
      if usineName.isalpha():
         Z = usineToZ[usineName.upper()]
         for z in Z:
            for k in nuclei.keys():
               if nuclei[k]['Z'] == z and not nuclei[k]['A'] == 0:
                  result.append(k)
      else:
         A = re.findall(r'\d+', usineName)[0]
         Z = usineToZ[re.findall(r'[A-Z]+', usineName.upper())[0]]
         for z in Z:
            for k in nuclei.keys():
               if nuclei[k]['Z'] == z and nuclei[k]['A'] == int(A):
                  result.append(k)

   return result

# Returns the charge and number of nucleons for a given isotope name
# Leptons satisfy A = 0 for simplicity.
# nuclei is a globally defined variable, initialized when the likelihood logp
# function is first run.
#
# Parameters: iso (str) name of isotope, eg. 'Boron_10'
# Returns tuple of ints Z, A
def getZAfromIso(iso):
   return nuclei[iso]['Z'], nuclei[iso]['A']

# Calculates the mass in units GeV/c^2 for a particle of a given number of nucleons
# Variations between particles of identical A but different Z are treated as negligible.
# This table is sourced from GALPROP v57, which in turns sources it from NIST,
# see also https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl
#
# Parameters: A (int) number of nucleons
# Returns: mass in units GeV/c^2 (float)
def getMfromA(A):
   amu = 0.931494
   # referencing ./source/Particle.cc from GALPROP v57
   masses = [0.0005485799, 1.007825, 2.014101, 3.016, 4.0026, 5.0, 6.015, 7.016, \
             8.0, 9.012, 10.012, 11.009, 12.0, 13.003, 14.003, 15.0, 15.995, \
             16.999, 17.999, 18.998, 19.992, 20.994, 21.991, 22.99, 23.985, \
             24.986, 25.982, 26.982, 27.977, 28.976, 29.973, 30.974, 31.972, \
             32.971, 33.968, 34.969, 35.967, 36.965, 37.963, 38.963, 39.963, \
             40.962, 41.959, 42.959, 43.956, 44.956, 45.953, 46.952, 47.95, \
             48.948, 49.946, 50.944, 51.941, 52.941, 53.939, 54.938, 55.935, \
             56.935, 57.934, 58.933, 59.931, 60.931, 61.928, 62.93, 63.929, \
             64.928, 65.926, 66.927, 67.925, 68.926, 69.924, 70.925, 71.922, \
             72.923, 73.922, 74.922, 75.92, 76.92, 77.918, 78.918, 79.916, \
             80.916, 81.915, 82.914, 83.912, 84.912, 85.91, 86.909, 87.906, \
             88.906, 89.904, 90.906, 91.906, 92.906, 93.906, 94.906, 95.906, \
             96.906, 97.906, 98.906]

   if A < 0 or A >= len(masses):
      logMessage(f'Invalid mass number {A} converted to M', 0)
      return 1

   return amu*masses[A]

# Returns a list of isotope names that must be summed up to obtain the numerator
# of quantityName. eg. getNumIsoList('B/C') = ['Boron_10', 'Boron_11']
# Sums are supported, eg. getNumIsoList('ELECTRONS+POSITRONS')
# We split on the first slash, meaning compound ratios like B/O / F/Si don't work
# See usineToFitsName() for more info
#
# Parameters: quantityName (str) name of the quantity in USINE format
# Returns: List of str, isotopes composing the numerator
def genNumIsoList(quantityName):

   numerator = quantityName.split('/')[0]
   listOfThings = numerator.split('+')
   result = []

   for l in listOfThings:
      result.extend(usineToFitsName(l))

   return result

# Returns a list of isotope names that must be summed up to obtain the denominator
# of quantityName. eg. getDenIsoList('B/C') = ['Carbon_12', 'Carbon_13']
# Sums are supported, eg. getDenIsoList('POSITRONS/ELECTRONS+POSITRONS')
# We split on the first slash, meaning compound ratios like B/O / F/Si don't work
# See usineToFitsName() for more info
#
# Parameters: quantityName (str) name of the quantity in USINE format
# Returns: List of str, isotopes composing the denominator, [] if not ratio
def genDenIsoList(quantityName):

   if '/' not in quantityName:
      return []

   denominator = quantityName.split('/')[1]
   listOfThings = denominator.split('+')
   result = []

   for l in listOfThings:
      result.extend(usineToFitsName(l))

   return result

# Finds a variety of general information from the FITS file, including the position
# of the sun, the bins in the R and Z directions, the energy bins, and the raw
# data. This is done here to avoid unnecessary opening of the FITS file.
# FITS file name is inferred from the results path and the galdef name.
#
# Parameters: None
# Returns: Tuple of numpy arrays containing the info listed above
#
# Note: Energy bins are returned in units of GeV/nuc, but the raw data are
# returned in units of (cm^2 sr s MeV^-1)^-1, so be mindful of this.
# GALPROP raw data are spectra multiplied by E**2
def procFITSHeader():

   fits_name = f'{results_path}/nuclei_57_{galdef_name}'

   hdul = fits.open(fits_name)

   R_sun = hdul['PRIMARY'].header['RSUN']

   Rbins_raw = hdul['GAL-R']
   Rbins = np.zeros(Rbins_raw.header['NAXIS2'])
   for (i,R) in enumerate(Rbins_raw.data):
      Rbins[i] = R[0]

   Zbins_raw = hdul['GAL-Z']
   Zbins = np.zeros(Zbins_raw.header['NAXIS2'])
   for (i,Z) in enumerate(Zbins_raw.data):
      Zbins[i] = Z[0]

   Ebins_raw = hdul['Energy']
   Ebins = np.zeros(Ebins_raw.header['NAXIS2'])
   for (i,E) in enumerate(Ebins_raw.data):
      Ebins[i] = E[0]*(1e-3)

   Etype = hdul['PRIMARY'].header['CUNIT3']
   if not 'MeV/nuc' in Etype:
      logMessage(f'Non-standard units {Etype} detected in FITS file.', 1)
      logMessage('Extraction will proceed but numerical results may not be correct', 1)

   data_raw = hdul['PRIMARY'].data
   hdul.close()

   return (R_sun, Rbins, Zbins, Ebins, data_raw)

# read the spectrum of the isotope iso from the raw data provided in headerInfo.
# headerInfo is essentially expected to just be the output of procFITSHeader()
# This extracts a single isotope, and will not work on eg. iso='Boron'
# The spectrum is unmodulated, ie LIS
#
# Parameters: iso (str) name of isotope whose spectra is requested
#    headerInfo (tuple of numpy arrays) see output of procFITSHeader()
# Returns: tuple of two numpy arrays, the first the energy bins and the second
#    the energy spectrum of the particle. Units are GeV/nuc and (m^2 sr s GeV)^-1
def readSpectraFITS(iso, headerInfo):

   global nuclei

   R_sun = headerInfo[0]
   Rbins = headerInfo[1]
   Zbins = headerInfo[2]
   Ebins = headerInfo[3]
   data_raw = headerInfo[4]

   Z_sun = 0 # assume sun is in galactic midplane by default

   # R_critical_L and R_critical_R are the indices of the R-bins that bound
   # R_sun below and above, respectively
   # weight_RL and weight_RR are used for linear interpolation to find the
   # spectrum precisely at R = R_sun, if R_sun is not one of the R bins
   R_critical_L = -1
   R_critical_R = -1
   weight_RL = -1
   weight_RR = -1
   for (i,R) in enumerate(Rbins):
      if i+1 >= Rbins.size:
         logMessage('ERROR: Failed to bound R_sun during header processing\n', -1)
         break
      elif R < R_sun and Rbins[i+1] >= R_sun:
         R_critical_L = i
         R_critical_R = i+1
         weight_RR = (R_sun - R) / (Rbins[i+1] - R)
         weight_RL = (Rbins[i+1] - R_sun) / (Rbins[i+1] - R)
         break

   Z_critical_L = -1
   Z_critical_R = -1
   weight_ZL = -1
   weight_ZR = -1

   if len(Zbins) > 1:
      for (i,Z) in enumerate(Zbins):
         if i+1 >= Zbins.size:
            logMessage('ERROR: Failed to bound Z_sun during header processing\n', -1)
            break
         elif Z < Z_sun and Zbins[i+1] >= Z_sun:
            Z_critical_L = i
            Z_critical_R = i+1
            weight_ZR = (Z_sun - Z) / (Zbins[i+1] - Z)
            weight_ZL = (Zbins[i+1] - Z_sun) / (Zbins[i+1] - Z)
            break
   else:
      Z_critical_L = 0
      Z_critical_R = 0
      weight_ZL = 0
      weight_ZR = 1

   if not iso in nuclei.keys():
      logMessage(f'Isotope name {iso} not recognized, skipping.', 0)
      return Ebins, np.zeros(Ebins.size)

   # multiplication by Ebins**-2 to remove the energy multiplication from raw data
   # multiplication by 10 to convert to standard units
   # (m^2 sr s GeV)^-1 = (100^2 cm^2 sr s 1000 MeV)^-1 = 10^7 (cm^2 sr s MeV)^-1
   # 10^7 (cm^2 sr s MeV)^-1 = 10 (cm^2 sr s MeV^-1)^-1 * 10^6 MeV^-2
   # = 10 (cm^2 sr s MeV^-1)^-1 * GeV^-2
   return Ebins, \
            ((data_raw[nuclei[iso]['index'], :, Z_critical_L, R_critical_L]*weight_RL*weight_ZL) + \
             (data_raw[nuclei[iso]['index'], :, Z_critical_R, R_critical_L]*weight_RL*weight_ZR) + \
             (data_raw[nuclei[iso]['index'], :, Z_critical_L, R_critical_R]*weight_RR*weight_ZL) + \
             (data_raw[nuclei[iso]['index'], :, Z_critical_R, R_critical_R]*weight_RR*weight_ZR)) * (Ebins**-2) * (10)

# apply solar modulation on the LIS theoryY_IS, with energy bins theoryX_IS
# Currently two modInfo options are supported:
# 'IS' : just return, interpolated to xbins if needed
# 'FF' : Use the force-field approximation to modulate, getting the phi parameter
#    from pvs as needed, and then interpolate to xbins if needed
#    See L. Gleeson and W. Axford, ApJ 154 p.1011 (1968) for details
#    See also C. Corti et al., PoS(ICRC2019)1070 (2019)
# Future releases may include further options
#
# Parameters: theoryX_IS (numpy array) the energy bins of the LIS
#    theoryY_IS (numpy array) the LIS aligned to theoryX
#    iso (str) name of the isotope / particle species
#    modInfo (str) option to choose which type of modulation to use. This is
#       generally provided in the Dataset object.
#    expName (str) name of the experiment to allow the correct phi to be chosen
#    pvs (dict) dictionary of parameter values, containing at minimum the relevant
#       phi values and their associated experiment names (phi should be given in MV)
#    xbins (numpy array, opt) optional argument for energy bins to align
#       the final result to through interpolation
# Returns: tuple of numpy arrays, the modulated spectrum and the energy bins it's
#    aligned to
def solar_modulate(theoryX_IS, theoryY_IS, iso, modInfo, expName, pvs, xbins=[]):
   Z, A = getZAfromIso(iso)
   if len(xbins) == 0:
      xbins = theoryX_IS

   if modInfo == 'IS':
      return xbins, interpolate(theoryX_IS, theoryY_IS, xbins)

   elif modInfo == 'FF':
      Z, A = getZAfromIso(iso)

      pmass = 0.939
      emass = 5.11e-4
      mass = emass if A == 0 else pmass
      A = 1 if A == 0 else A

      phi = 0
      logMessage(f'Parameters {str(pvs)} passed in', 3)
      for p in pvs.keys():
         if p in interfaceInfo.keys():
            if interfaceInfo[p]['type'] == 'modulation_FF':
               if expName in interfaceInfo[p]['expnames']:
                  phi = pvs[p]

      T_hp = xbins + ((np.abs(Z)/A)*phi*1e-3)
      data_phi = interpolate(theoryX_IS, theoryY_IS, T_hp) * (xbins*(xbins + 2*mass))/(T_hp*(T_hp + 2*mass))

      return xbins, data_phi

   logMessage(f'Unknown solar modulation type {modInfo}', -1)

# convert a spectrum theoryY and energy bins theoryX from kinetic energy per nucleon
# to either rigidity, kinetic energy, or total energy as requested.
# Input units are expected to be GeV/nuc and (m^2 sr s GeV)^-1 respectively.
#
# Note: the reason we convert models to data units and not the other way around
# is because we know the precise isotopic composition of the model, but may not
# know this for the data. To avoid biases we convert the model using exact A and M
# values. See L. Derome et al., A&A 627, A158 (2019) Appendix A for details
#
# Parameters: theoryX (numpy array) energy bins, units GeV/nuc
#    theoryY (numpy array) Spectrum aligned to theoryX, units (m^2 sr s GeV)^-1
#    iso (str) name of isotope / particle species
#    unit (str) one of 'EKN', 'R', 'EK', or 'ETOT'
# Returns: tuple of 2 numpy arrays containing the converted spectrum and bins
def matchUnits(theoryX, theoryY, iso, unit):

   Z, a = getZAfromIso(iso)

   M = getMfromA(a)
   A = 1 if a == 0 else a

   if unit == 'EKN':
      return theoryX, theoryY
   elif unit == 'R':

      Etot = (A*theoryX + M)
      theoryX_conv = (1/np.abs(Z))*np.sqrt(Etot**2 - M**2)
      theoryY_conv = theoryY * (np.abs(Z)/A) * (np.sqrt(Etot**2 - M**2) / Etot)

      return theoryX_conv, theoryY_conv
   elif unit == 'EK':

      theoryX_conv = A*theoryX
      theoryY_conv = theoryY/A
      return theoryX_conv, theoryY_conv
   elif unit == 'ETOT':

      theoryX_conv = A*theoryX + M
      theoryY_conv = theoryY/A
      return theoryX_conv, theoryY_conv

   logMessage(f'Conversion attempted to unknown unit {unit}', -1)
   return 0

# align realY, originally aligned to realX, to newX using power-law interpolation
# specifically we fit a power law to the nearest points of realY above and below each newX
# value and then interpolate on that power law to produce newY.
# Non-positive y values and extrapolation are not supported and will throw NaN
#
# Parameters: realX (numpy array) old bins
#    realY (numpy array) old values aligned to realX
#    newX (numpy array) new bins
# Returns numpy array of values aligned to newX
def interpolate(realX, realY, newX):

   newY = np.zeros(newX.size)
   newY[:] = np.nan
   c = 0
   for (i, x) in enumerate(newX):
      while c+1 < realX.size and realX[c+1] < x:
         c += 1

       # handles the case where we run off the end of realX without finding a place to interpolate
      if c+1 >= realX.size or realX[c] > x:
         logMessage('Interpolation attempted beyond scope of real X data, skipping', 2)
         logMessage(f'Point {x} outside bounds of real X data', 3)
         continue

      if realY[c+1] <= 0 or realY[c] <= 0:
         logMessage('Non-positive y-value detected in log interp, skipping', 1)
         logMessage(f'Points ({realX[c]}, {realY[c]}) and ({realX[c+1]}, {realY[c+1]})', 3)
         continue

      if realX[c+1] <= 0 or realX[c] <= 0:
         logMessage('Non-positive x-value detected in log interp, skipping', 1)
         logMessage(f'Points ({realX[c]}, {realY[c]}) and ({realX[c+1]}, {realY[c+1]})', 3)
         continue

      # y = b*(x^g) is assumed, true for cosmic ray spectra usually
      g = np.log(realY[c+1]/realY[c]) / np.log(realX[c+1]/realX[c])
      b = realY[c] / (realX[c]**g)
      newY[i] = b*(x**g)

   return newY

# modulate the LIS theoryY_IS and align it to xbins with interpolation.
# The purpose of this function is to remove unnecessary interpolations and improve
# the accuracy of model calculations and comparisons to the data.
# We accomplish this by passing in the bins of the experimental data to xbins
# and interpolating to those bins during the modulation calculation, which generally
# involves interpolating no matter what.
# Parameters: see solar_modulate() for most parameter definitions.
#    unit (str) the units of xbins, either 'EKN', 'R', 'EK', or 'ETOT'
#    note xbins is not optional here, but is optional in solar_modulate()
# Returns: modulated spectrum, aligned to xbins (converted to kinetic energy per nucleon)
def modAndInterp(theoryX_IS, theoryY_IS, iso, modInfo, expName, pvs, unit, xbins):
   finalXbins = xbins
   Z, a = getZAfromIso(iso)

   M = getMfromA(a)
   A = 1 if a == 0 else a
   # xbins could be in any unit, but solar_modulate() only accepts EKN so convert
   if unit == 'R':
      finalXbins = (np.sqrt((np.abs(Z)*finalXbins)**2 + M**2) - M) / A
   elif unit == 'EK':
      finalXbins = xbins/A
   elif unit == 'ETOT':
      finalXbins = (xbins - M)/A

   return solar_modulate(theoryX_IS, theoryY_IS, iso, modInfo, expName, pvs, xbins=finalXbins)

# calculate the chi-squared (sum of squared residuals) against the Dataset d using
# the parameters pvs. The process is to sum across all isotopes for the numerator
# and denominator, then take the ratio and sum the squared residuals. For each
# isotope we read the LIS, modulate and interpolate to the data bins, and match
# the units from GeV/nuc to whatever the data is in. This approach minimizes
# bias in the calculation due to unit conversion and interpolation, see matchUnits()
# and modAndInterp() for details.
#
# Parameters: d (Dataset) Dataset object that we want the chi-squared for
#    pvs (dict) dictionary of parameter values
#    headerInfo (tuple) output of procFITSHeader()
# Returns: chi-squared of the model against d (float)
def calculateChiData(d, pvs, headerInfo):

   quantityName = d.getQuantityName()
   numeratorIsotopeList = genNumIsoList(quantityName)
   denominatorIsotopeList = genDenIsoList(quantityName)

   modInfo = d.modulationInfo
   expName = d.expName
   unit = d.unit
   xBins = d.xVals
   errs = np.ones(len(xBins))
   upLim = d.upLim

   logMessage(f'Entering chi calc for dataset {quantityName} {expName}', 3)

   logMessage(f'num is {numeratorIsotopeList}', 3)
   logMessage(f'denom is {denominatorIsotopeList}', 3)

   # these arrays store the summed spectra as they're calculated
   numeratorY = np.zeros(len(xBins))
   denominatorY = np.zeros(len(xBins))

   for n in numeratorIsotopeList:
      theoryX_IS, theoryY_IS = readSpectraFITS(n, headerInfo)

      #theoryX, theoryY = solar_modulate(theoryX_IS, theoryY_IS, n, modInfo, expName, pvs)
      #theoryX_conv, theoryY_conv = matchUnits(theoryX, theoryY, n, unit)
      #theoryY_final = interpolate(theoryX_conv, theoryY_conv, xBins)

      theoryX, theoryY = modAndInterp(theoryX_IS, theoryY_IS, n, modInfo, expName, pvs, unit, xBins)
      theoryX_final, theoryY_final = matchUnits(theoryX, theoryY, n, unit)

      numeratorY += theoryY_final

   for D in denominatorIsotopeList:
      theoryX_IS, theoryY_IS = readSpectraFITS(D, headerInfo)

      #theoryX, theoryY = solar_modulate(theoryX_IS, theoryY_IS, D, modInfo, expName, pvs)
      #theoryX_conv, theoryY_conv = matchUnits(theoryX, theoryY, D, unit)
      #theoryY_final = interpolate(theoryX_conv, theoryY_conv, xBins)

      theoryX, theoryY = modAndInterp(theoryX_IS, theoryY_IS, D, modInfo, expName, pvs, unit, xBins)
      theoryX_final, theoryY_final = matchUnits(theoryX, theoryY, D, unit)

      denominatorY += theoryY_final

   # take ratio if ratio data
   # numeratorY contains the final result no matter what
   if len(denominatorIsotopeList) > 0:
      numeratorY = numeratorY / denominatorY

   logMessage(f'Final xBins: {xBins}', 3)
   logMessage(f'Final theory curve: {numeratorY}', 3)
   logMessage(f'Data yvals: {d.yVals}', 3)

   # we want to support positive and negative error bars being different sizes,
   # as well as upper limit data. We do that here, selecting the correct error
   # bar to be used in the chi-squared calculation for each point
   for (i,e) in enumerate(errs):
      if numeratorY[i] > d.yVals[i]:
         errs[i] = d.posYErr[i]
      else:
         if upLim[i] == 1:
            errs[i] = np.inf
         else:
            errs[i] = d.negYErr[i]

   logMessage(f'Data errs: {errs}', 3)

   # the chi-squared is assumed to be simply the sum of the squared residuals.
   # This of course does not apply if the data or errors are correlated
   # We do not support correlations in data as of right now
   # Model uncertainties are also ignored
   summed = sum(((numeratorY - d.yVals)/errs)**2)

   logMessage(f'Residuals: {((numeratorY - d.yVals)/errs)**2}', 3)
   logMessage(f'Chi {summed} for dataset {quantityName} {expName}', 3)

   return summed

# ==================================
# CLASS DEFINITIONS
# ==================================

# The Dataset class contains information about experimental datasets. It is
# intended to contain data only from a single experiment and a single quantity.
# Mostly self-explanatory, Datasets are pretty easily generated from USINE files
class Dataset:

   # combine the numerator and denominator into a single string, if necessary
   #
   # Parameters: None
   # Returns: quantity name (str)
   def getQuantityName(self):
      if self.denominator is not None:
         return self.quantity + '/' + self.denominator
      else:
         return self.quantity

   # initialization is performed using an entry from a dictionary generated by
   # generateUsineDict(). See ListDatasetsUsine() for more info
   #
   # Parameters: key (str) key assigned to the entry in the usineDict
   #    usinedictentry (dict) the value of the usineDict[key]
   #    modulationInfo (str) option for how to modulate this Dataset
   # Returns: None
   def __init__(self, key, usinedictentry, modulationinfo):

      self.unit = str(usinedictentry['Unit'])
      self.quantity = key[0].split('/')[0]
      self.denominator = None
      if '/' in key[0]:
         self.denominator = key[0].split('/')[1]
      self.expName = str(key[1])
      self.modulationInfo = modulationinfo

      self.xVals = usinedictentry['Ebins']
      self.yVals = usinedictentry['Value']
      self.negYErr = usinedictentry['Err-']
      self.posYErr = usinedictentry['Err+']
      self.upLim = usinedictentry['upLim']

# Theory code corresponding to GALPROP, meaning solar modulation is not included
# in any way. Currently, any parameter that is not a modulation parameter is
# needed as input, and the error code of GALPROP and the nuisance chi-squared
# contribution from non-modulation parameters are provided.
class GalpropDriver(Theory):

   def initialize(self):
      return None

   # we don't need any modulation parameters, but currently we do need all the
   # rest (Xsec, galdef, and source)
   # This optimization allows for more efficient sampling, eg. using fast-dragging
   def get_requirements(self):
      reqs = {}
      for p in params:
         if 'modulation' not in interfaceInfo[p]['type']:
            reqs[p] = None

      return reqs

   def get_can_provide(self):
      return ['galperrcode', 'galpnuisancechi']

   # there are 3 steps in running GALPROP: resetting the eval_iso_cs.dat file
   # which contains the production cross-sections for several important channels,
   # setting the parameters, and then running GALPROP and returning the error
   # code as well as the nuisance parameter chi-squared contribution from any
   # non-modulation parameters.
   def calculate(self, state, want_derived=False, **params_values_dict):
      if os.path.exists(f'{galtools_share_path}/eval_iso_cs_orig1.dat'):
         logMessage('Resetting eval_iso_cs.dat to original values', 3)
         reset_prod_Xsec_cmd = f'cp {galtools_share_path}/eval_iso_cs_orig1.dat {galtools_share_path}/eval_iso_cs.dat'
      else:
         logMessage('Making a copy of eval_iso_cs.dat for resetting purposes', 3)
         reset_prod_Xsec_cmd = f'cp {galtools_share_path}/eval_iso_cs.dat {galtools_share_path}/eval_iso_cs_orig1.dat'

      reset_prod_Xsec = subprocess.Popen([reset_prod_Xsec_cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
      o, e = reset_prod_Xsec.communicate()

#      for p in self.get_requirements():
#         setParameter(p, params_values_dict[p])
      setParameters(params_values_dict)

      cmd = f'{galprop_path} -f {FITS_path} -g {galdef_path} -o {results_path} -r {galdef_name}'
      galp = subprocess.Popen([cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

      o, e = galp.communicate()
      output = e.decode('ascii')
      logMessage(f'GALPROP output (errstream): {output}', 3)
      logMessage(f'GALPROP error code: {str(galp.returncode)}', 3)

      state['galperrcode'] = galp.returncode
      state['galpnuisancechi'] = 0
      for p in self.get_requirements():
         if "nuisance" in interfaceInfo[p].keys():
            if interfaceInfo[p]["nuisance"]:
               state['galpnuisancechi'] += ((params_values_dict[p] - interfaceInfo[p]["mu"])/interfaceInfo[p]["sigma"])**2

      return True



class GalpropLikelihood(Likelihood):

   def initialize(self):
      self.data = listDatasetsUsine(dataUsine_filepath)
      self.numSets = len(self.data)
      self.provider = GalpropDriver

   def get_requirements(self):

      reqs = {}
      for p in params:
         # currently modulation_FF is the only type that doesn't go to GALPROP
         if 'modulation' in interfaceInfo[p]['type']:
            reqs[p] = None

      reqs['galperrcode'] = None
      reqs['galpnuisancechi'] = None
      return reqs

   def logp(self, **params_values):

      global nuclei

      err = self.provider.get_result('galperrcode')
      if not err == 0:
         logMessage(f'GALPROP failed with error code {err}, exiting', -1)

      # pre-populate nuclei once so it doesn't have to be recomputed again
      # we assume the list of nuclei propagated doesn't change in a single run
      # FITS file should always be present since GALPROP is run once above this
      if len(nuclei.keys()) == 0:
         fits_name = f'{results_path}/nuclei_57_{galdef_name}'
         hdul = fits.open(fits_name)
         nuclei_raw = hdul['NUCLEI']
         for (i,N) in enumerate(nuclei_raw.data):
            nuclei[N[0]] = {'index' : i,
                            'Z'     : N[1],
                            'A'     : N[2]}
         hdul.close()

      # set modulation, etc. parameters
      setParameters(params_values)

      chis = np.zeros(self.numSets)
      # headerInfo re-processed each iteration in case dz, dr, etc. are varied
      headerInfo = procFITSHeader()
      for (i,d) in enumerate(self.data):
         chis[i] = calculateChiData(d, params_values, headerInfo)

      chi = sum(chis)
      logMessage(f'Chi pre-nuisance: {chi}', 3)

      # receive the nuisance chi terms from the GALPROP parameters
      chi += self.provider.get_result('galpnuisancechi')

      # additionally calculate nuisance chi terms for non-GALPROP parameters
      for p in self.get_requirements():
         if p in interfaceInfo.keys():
            if "nuisance" in interfaceInfo[p].keys():
               if interfaceInfo[p]["nuisance"]:
                  chi += ((params_values[p] - interfaceInfo[p]["mu"])/interfaceInfo[p]["sigma"])**2

      logMessage(f'Chi: {chi}', 3)

      return -chi / 2

# ==================================
# OPTIMIZATION SETUP
# ==================================

"""
info = {
    "likelihood": {'galpLikelihood' : GalpropLikelihood},
    "theory" : {'GalpropDriver' : GalpropDriver},
    "params": dict([
        ("D_0", {
            "prior": {"min": 1.0e28, "max": 20.0e28},
            "ref" : {"min": 6.3e28, "max": 6.4e28},
            "proposal" : 0.1e28,
            "latex": r"D_0"}),
        ("delta", {
            "prior": {"min": 0.0, "max": 1.0},
            "ref" : {"min": 0.350, "max": 0.370},
            "proposal" : 0.003,
            "latex": r"\delta"}),
        ("v_a", {
            "prior": {"min": 0.0, "max": 50.0},
            "ref" : {"min": 29.0, "max": 33.0},
            "proposal" : 0.5,
            "latex": r"v_{Alfven}\ (km/s)"}),
        ("gamma_0", {
            "prior": {"min": 0.0, "max": 3.0},
            "ref" : {"min": 1.8, "max": 1.9},
            "proposal" : 0.01,
            "latex": r"\gamma_0"}),
        ("gamma_1", {
            "prior": {"min": 0.0, "max": 3.0},
            "ref" : {"min": 2.3, "max": 2.4},
            "proposal" : 0.005,
            "latex": r"\gamma_1"}),
        ("ab_12C", {
            "prior": {"min": 0, "max": 10000},
            "ref" : {"min": 3200, "max": 3400},
            "proposal" : 20,
            "latex": r"ab^{12}C"}),
        ("ab_16O", {
            "prior": {"min": 0, "max": 10000},
            "ref" : {"min": 4400, "max": 4500},
            "proposal" : 20,
            "latex": r"ab^{16}O"}),
        ("phi", {
            "prior": {"min": 0, "max": 1500},
            "ref" : {"min": 450, "max": 550},
            "proposal" : 10,
            "latex": r"\phi\ (MV)"}),
        ]),
#    "sampler": {"mcmc" : {'drag' : True, 'oversample_power' : 0.75, 'measure_speeds' : 3}},
    "sampler": {"evaluate" : None},
    "output": "speedtestSep6"}

interfaceInfo = dict([
    ("D_0", {
        "type" : 'galdef',
        "galpName" : 'D0_xx'}),
    ("delta", {
        "type" : 'galdef',
        "galpName" : 'D_g_2'}),
    ("v_a", {
        "type" : 'galdef',
        "galpName" : 'v_Alfven'}),
    ("gamma_0", {
        "type" : 'source',
        "galpName" : 'spectral_pars',
        "position" : 0}),
    ("gamma_1", {
        "type" : 'source',
        "galpName" : 'spectral_pars',
        "position" : 2}),
    ("ab_12C", {
        "type" : 'source',
        "galpName" : 'iso_abundance_06_012'}),
    ("ab_16O", {
        "type" : 'source',
        "galpName" : 'iso_abundance_08_016'}),
    ("phi", {
        "type" : 'modulation_FF',
         "expnames" : ['AMS02(2011/05-2016/05)']}),
    ])
"""

info = yaml_load_file(cobaya_yaml_filepath)
interfaceInfo = yaml_load_file(interface_yaml_filepath)

params = list(info["params"].keys())
n_params = len(params)

# used for profiling the code's efficiency, disabled
# cProfile.run("run(info)")

updated_info, sampler = run(info, resume=True)
