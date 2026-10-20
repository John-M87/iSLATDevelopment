# Changelog

## v5.03.04 - October 20, 2026

### New Features

#### Plot Views & Y Axis
- Added Y-axis controls to the `Views` menu in the top bar: an `Auto-scale Y Axis` toggle, an `Auto-scale Y to Model Flux` toggle (fits the axis to the plotted models instead of the observed data), a `Set Y Limits…` dialog for pinning a manual Y range, and a `Show Observed Data` toggle
- Added new options to the Center View function (center on a wavelength, display an exact wavelength range, or display a fixed number of data points) and added a `Center View… (C)` entry to the Spectral Functions menu
- Added the ability to bold a molecule's model line in the plot via a new `Bold on Plot` / `Unbold on Plot` option in the molecule right-click menu
- Updated the population diagram plot to make better use of the quantum state schema module

#### Line List Filtering
- Added a new `LineFilterExpression` module for filtering line lists with composable condition groups (value ranges, derived quantities, velocity separation, quantum fields/labels, species) that can be saved to and loaded from JSON
- Added a new `FilteredPopulationDiagramWindow` for displaying population diagrams of filtered line lists
- Updated `LineListMaker` to support filtering based on expressions (`filter_expression`, `filter_any`, `filter_all`, `filter_quantum_field`, `filter_vib_band`)
- Added a velocity separation filter and percentile-based filtering to the line list filter window
- Added the ability to filter by either percentile rank or percentage of the largest value
- Added `Save Line List…` and `Load Line List…` options to the line list filter window
- The line list viewer window can now display the currently loaded line list

#### Export & Bulk Apply
- Added the optical depth as a function of wavelength to the CSV output of the Export Models function
- Added the ability to set the file name/location in the Export Models popup window; exports now use the `MoleculeDict` match pixel sampling when writing a model to CSV
- Added the instrumental profile to the Bulk Apply window
- The Bulk Apply window now lets you specify how the properties should be applied (e.g. to visible molecules only)

### Bug Fixes
- Fixed a bug with the model range extension in the spectrum plotting modules: models are now drawn over the full global wavelength range even when `Match Pix. Sampling` resamples them onto a narrower observed grid
- Fixed issues with the dedicated Line Inspection plot view
- Fixed incorrect line picking when multiple molecules are plotted in the line inspection plot
- Fixed a memory leak in the `PopulationDiagramView`
- Fixed the `_compute_intensity_hash` method of the `Molecule` class to use the `INTENSITY_AFFECTING_PARAMS` attribute instead of a hardcoded list of parameters
- Added `model_pixel_res` to the molecule's `SPECTRUM_AFFECTING_PARAMS` list so that pixel resolution changes correctly invalidate cached spectra

### Code Quality
- Changed the weak blend threshold in the `Intensity` class from 4.0 to 0.01
- The `Intensity` line info output now reports line center flux, line center opacity, and model opacity in range
- Updated the population diagram examples notebook to use the new method of setting the active molecule in the grid
- Added tests for the filtered population diagram window, line filter expression module, center view, and model range extension
- Bumped version to `v5.03.04`

---

## v5.03.03 - August 11, 2026

### New Features

#### GUI Improvements
- Added the ability to sort molecules in the control panel by their properties (new `Sort Molecules` option in the Molecules menu)
- Added a new `Bulk Apply Properties` popout window to the Molecules menu for applying parameter values to many molecules at once
- Added `N Cols` and `N Rows` fields to the control panel when in the Line Grid view
- Updated the Sample Manager window: the loaded sample is now saved to `SampleConfig.json` and can be restored on startup via the new `restore_sample_on_startup` user setting (user settings version bumped to 1.4)
- Clicking on the text label of a line in the line inspection view now also highlights the line
- Added a match sampling toggle to the Residual Spectrum Plot module

#### Keyboard Shortcuts
- Added the `C` key to open the Center View dialog for centering the GUI on a specific wavelength
- Holding `Shift` while advancing or retreating the plot start (`>` / `<`) now moves by half of the current plot range

#### Data
- Updated the list of available HITRAN isotopologues (additional SO2, NO2, OCS, HCOOH, SO and CH3F isotopologues, plus H3+, CH3, S2, COFCl, HONO and ClNO2). Note: the new entries will not work until the `hitran` module of `astroquery` is updated to include them

### Bug Fixes
- Fixed various bugs with the Line Grid view
- Fixed model lines not honoring the pixel sampling setting in the main plot grid
- The Save Line function now uses the actual selection range (`current_selection`) instead of the nearest pixels in the wavelength array (`selected_wave`)
- Fixed an issue with the loading order of the Manage Sample function that caused the control panel to get out of sync

### Code Quality
- Added `SampleConfig.json` to `.gitignore`
- Bumped version to `v5.03.03`

---

## v5.03.02 - June 09, 2026

### New Features

#### GUI Improvements
- Added a new iSLAT logo, now used as the application window icon (WIP)
- Added the ability to drag and drop molecules in the control panel to reorder them
- Added a new `View Line List` window (with search, refresh and `Export CSV…`) opened from the molecule right-click menu
- Added a `Download and Add to iSLAT` button to the HITRAN query window
- Added a new merge molecules function to the GUI and `MoleculeDict` (`Merge Selected` in the molecule right-click menu)
- Added a WIP `SampleManagerWindow` to the Spectral Functions menu (`Manage Sample…`)
- Updated scrolling to work across platforms
- Plot start/range values are now displayed with general (`g`) formatting at 4 significant digits, with increased precision

#### Population Diagram
- Added percentile and exact value options to the population diagram axis settings
- Added a WIP `Shape By` option to the population diagram axis settings menu

#### Keyboard Shortcuts
- Changed the keybind for advancing and retreating the plot start to `>` / `<` instead of `N` / `Shift+N`

#### Plotting API
- The model line width and alpha used by the `SpectrumPanel` generate plot method are now configurable via kwargs, and model lines are more visible in the three panel view

### Bug Fixes
- Fixed a bug where the `Molecule` class would use the wrong pixel resolution when calculating its flux
- Fixed a bug in the `FullSpectrumPlot` where setting the wave min/max overrode the plot range set from the GUI
- The save figure function of the `FullSpectrumView` now uses the wave min/max values from the GUI
- Fixed the legend toggle not syncing, and fixed the toggle legend state not being respected when a molecule's visibility is updated

### Code Quality
- Updated comment/docstring formatting in several files
- Bumped version to `v5.03.02`

---

## v5.03.01 - May 27, 2026

### Plotting System Refactor
- Completed the migration of the GUI plotting onto the view-based architecture: the `PlotRenderer` class has been deleted and `MainPlot` now delegates rendering to the active `PlotView`
- Added a new `CompositePlot` abstract base class; `MainPlotGrid`, `FitLinesPlotGrid` and `StackedSpectralPanel` now inherit from it
- Moved fit rendering from `MainPlotGrid` to the `SpectrumPanel` class, and moved active-line handling onto `BasePlot` (implemented by `LineInspectionPlot` and `PopulationDiagramPlot`)
- Moved the flux integral calculation from `LineAnalyzer` to a new `spectral_utils` module
- Moved line property labels and names into the `MoleculeLine` class
- The `Spectrum` class now owns the Keplerian FWHM

### New Features

#### Plot Views
- Added a new `Views` drop-down menu to the top bar for switching between plot views
- Added new `PopulationDiagramView`, `LineInspectionView`, `SpectrumPanelView` and `FitLinesPlotGridView` ("Line Grid") views
- Views now control their own right-click context menus; added `Switch to …` options to move between views, a `Save Figure…` option on every view's context menu, and `Open in Full Spectrum` / `Open in Three Panel` interactions
- Added a toggle in the GUI to scale the full spectrum view by data density instead of wavelength
- The Line Grid view can toggle the visibility of model lines, and opening a line in the three panel view opens a span that encompasses that line
- Added the `T` key to toggle the three panel view

#### Population Diagram
- Added log scale and percentile cutoff options to the population diagram `Color By…` function, with suggested log scale fields in the color-by window
- Added an `Axis Settings…` dialog to the population diagram right-click menu, including FWHM and quantum-state-label axis options
- Shift-clicking any line on the population diagram now sets the line inspection plot to that line
- Added a `Save Active Lines as Line List…` option to the population diagram context menu

#### Line Inspection
- Added right-click context options to the line inspection plot: save/fit the current line, run the deblender, save all lines in range, hide/show the total model, match the spectrum to the inspection range, and center the spectrum on the inspection location

#### Quantum State Schemas
- Added new `QuantumStateSchema` and `HITRANQuantumSchemas` infrastructure for parsing HITRAN quantum state labels per molecule, used by the population diagram and line list filtering
- Added a new `PlotAxisRegistry` for axis and label specification

#### GUI Improvements
- Added a new `Import Parameters From File` function to the top bar
- Added `Change Line List…` and `Delete` options to the molecule right-click menu; the default active molecule can now be deleted from the control panel
- Removed the plus and x buttons from the `FileInteractionPane` and moved their functionality to the other buttons
- Removed the word "Toggle" from the toggle buttons so they take up less space in the GUI
- Made scrolling work correctly in the control panel

#### Fitting
- Widened the `Slabfit` parameter bounds (`n_mol` upper bound raised to 1e35, log radius upper bound raised to 8.0)

### Bug Fixes
- Fixed a bug where toggling a molecule's visibility ON in the spectrum panel did not plot the molecule if it had no existing artists
- Fixed summed flux not updating correctly in the three panel view
- Fixed `FullSpectrumPlot`/`FullSpectrumView` not updating with molecule parameter changes, and the residual spectrum plot not updating when a new spectrum is loaded
- Fixed residual spectrum plot models and chi² values not updating when control panel fields were changed, and fixed molecule flux still being counted in the residuals regardless of visibility
- Fixed the main plot grid not correctly using the global wavelength range value from the control panel
- Fixed x-axis labels overlapping in the full spectrum view
- Fixed the atomic lines toggle error and the full spectrum view molecule visibility toggle
- Fixed the toggle full spectrum button, click events in the `LineInspectionView`, and the toggle total model function across the different views
- Fixed an error when deleting a molecule that was created via the duplication function
- Fixed an error message from the tooltip code

### Code Quality
- Removed unused code from `MainPlot`, `BasePlot` and other plotting classes
- Updated the README with the Zenodo DOI and citation for version 5
- Bumped version to `v5.03.01`

---

## v5.03.00 - May 03, 2026

### New Features

#### Multiple Active Molecules
- Added the ability to select multiple active molecules; the active set is now managed by `MoleculeDict` (`active_molecule`, `comparison_molecules`, `get_active_set`, `toggle_comparison_molecule`, `promote_to_active_molecule`)
- The line inspection plot and its legend now render all active molecules
- Draft population diagram support for the current set of active molecules from within the UI

#### Instrumental Profiles
- Added the ability to set instrumental profiles for determining the resolving power in the control panel (`Constant FWHM` or `JWST MIRI MRS` wavelength-dependent R(λ)); profiles are defined in the new `InstrumentalProfiles` module
- Split the FWHM control into `Inst. FWHM` and `Kep. FWHM` fields (Keplerian broadening is added in quadrature with the instrumental profile)
- Added a new readout to the data field that displays the instrumental, Keplerian, and convolved FWHM

#### Molecule Right-Click Menu
- Added a new context menu that opens when a molecule name is right-clicked in the control panel, with `Export Model` and `Filter Line List` options
- Added a new `LineListFilterWindow` with tooltips, an `Apply to Molecule` button, and an export to `.par` button

#### Population Diagram & Line Inspection
- Refactored the population diagram plot for more flexible plotting options and added a new `Population_Diagram_Examples.ipynb` notebook
- Added a right-click context menu to the population diagram exposing the color-by functionality, including line center opacity as a color-by option
- Added right-click options to the line inspection plot

#### GUI Improvements
- Moved the plot step controls to the top bar instead of the control panel
- Reorganized the GUI menus and added tooltips to the drop-down menu functions
- Added a drop-down selection to the file interaction pane for switching between spectra when a sample is loaded
- The data field now extends to the bottom of the window automatically
- When a slab fit finishes, the control panel is now updated with the new parameter values
- Added new `ControlField`/`ControlSurface` abstractions that allow GUI components to register fields in various control locations

#### Fitting & Data Processing
- First draft of the `Slabfit` refactor (`SlabModel`, `FitParameter`, `FitResult`, with `from_molecule`/`from_intensity`/`from_line_list`/`from_spectrum` constructors)
- Reworked `MoleculeDict` flux summation to work more consistently with different molecule FWHMs/grid sizes

### Bug Fixes
- Fixed the population diagram rendering with all molecules on launch
- Fixed the line inspection threshold being ignored
- Fixed an issue with the `MoleculeDict` and `FullSpectrumView` residual toggle
- Fixed an incorrect saved lines toggle
- Fixed the active molecule being overridden when multiple molecules are selected
- Fixed an incorrect callback attribute check in the `ControlPanel`
- Fixed an issue with the Subtract Models from Data function and added tests for it

### Code Quality
- Refactored `MainPlot`, `MainPlotGrid` and `ThreePanelView` to make better use of the view and `BasePlot` based infrastructure; `ThreePanelView` now composes a `MainPlotGrid` in the same fashion as `FullSpectrumView`
- Removed old/unused code from the interaction handler, `iSLAT` class, `MainPlot` and `PlotRenderer`
- Bumped version to `v5.03.00`

---

## v5.02.05 - April 12, 2026

### Plotting Module Refactor
- Introduced a new abstract `StackedSpectralPanel` class that composes an arbitrary number of `SpectralPanel` objects, with concrete `SpectrumPanel` and `ResidualPanel` implementations
- Added a new `ResidualSpectrumPlot` class (a subclass of `FullSpectrumPlot`) with an additional residual sub-panel and chi² calculation per panel
- Added new `OpticalDepthPanel` and `OpticalDepthSpectrumPlot` classes for plotting τ(λ), with a new `Optical_Depth_Examples.ipynb` notebook
- Split the `CompositeStackedPanel` class (merged cells from two source plots) into its own file
- `FullSpectrumView` now works with any class that implements `StackedSpectralPanel`
- Added WIP shading for gaps in data (`GapMode`) and a WIP option to scale the x-axis by data density instead of wavelength (`XScaling`)
- Started a legend handling refactor (`LegendStrategy`, `StandardLegend`, `MoleculeColorLegend`)
- `PopulationDiagramPlot` can now be used with bare `Intensity` objects
- Improved `BasePlot` behaviour in notebooks (figure management and embedding)

### New Features

#### Data Types
- Added support for format-neutral line data sources in `Molecule`, `MoleculeDict` and `MoleculeLineList`: a `line_data_source` path plus a `line_format` (`hitran`, `csv`, or `saved`) handled by the new `line_list_readers` module
- Added an experimental wavelength-dependent resolving power to the `Spectrum` class, and a `reset` method for reusing a `Spectrum` object with different parameters
- Added additional batch calculation methods to the `Intensity` class
- `Slabfit` now exports results as JSON by default

#### GUI Improvements
- Experimental residual spectrum toggle in the full spectrum view (`R` key)
- Added a new `Load Parameters From File (Ctrl+Shift+L)` button to let the user select a specific parameter file to load
- Added new buttons/keybinds to advance or reverse the plot start by the current range value
- When a line list is not set when running a slab fit, the user is now prompted to select one
- Refactored the control panel and plot view classes to allow for dynamically created control panel fields

### Performance Improvements
- Reworked many calculations in the `Intensity` and `Spectrum` classes to use in-place operations, vectorized more operations, and reduced the number of temporary arrays created, especially in the blended line processing
- Added a unique-wavelength summarization step to the `Spectrum` class for when multiple intensity components share the same wavelength
- Added `wavelength_range` to the `Intensity`/`Molecule` hash functions to allow caching of intensity arrays for different wavelength ranges
- Improved the startup import call chain to avoid circular imports and speed up startup time

### Bug Fixes
- Fixed the population diagram plot having incorrect lines, and refactored the intensity table methods
- Fixed the full spectrum view theme not persisting
- Fixed annotations and gap detection when the gap mode is set to skip in the residual/full spectrum plots
- Fixed a bug when loading new data in the full spectrum / residual views
- Fixed the distance and stellar RV fields not being visible by default after the control panel changes, and fixed the display range field not syncing between views
- Fixed the attribute assignment for the `spectres` flag in `Molecule.py`
- Fixed a notebook embedding error and pyplot figure management in the `BasePlot` class

### Code Quality
- Added a `pytest` test suite (`tests/`) covering the data types, data processing, plotting and GUI modules
- Added an `__init__` file to the `Widgets` folder
- Updated the plotting example notebooks
- Bumped version to `v5.02.05`

---

## v5.02.04 - April 12, 2026

### New Features
- Updated the `Molecule` and `Spectrum` classes to use flux-conserving resampling in the same fashion as the match pixel sampling function

### Bug Fixes
- Fixed the toggle legend keybind to work correctly
- Fixed the file dialog not opening on top of other windows
- Disabled the data field output when the match spectral sampling button is toggled
- Fixed the `MainPlot` class clearing the line inspection plot when the model pixel resolution is updated

### Code Quality
- Bumped version to `v5.02.04`

---

## v5.02.03 - March 08, 2026

### Plotting System Refactor
- Major refactor of the plotting system to use a modular, view-based architecture with new `PlotView` classes for different plot layouts
- Introduced switchable plot views (`ThreePanelView`, `FullSpectrumView`) eliminating scattered conditional checks throughout the codebase
- Refactored `PlotRenderer` and related plotting classes for more flexible rendering of lines and improved inheritance structure
- Refactored the output full spectrum functionality into `FullSpectrumView`, eliminating the separate `OutputFullSpectrum` module
- Speed improvements to the plotting API
- Updated the full spectrum PDF output to use dynamic tick spacing based on panel width instead of hardcoded 0.25 μm
- Reformatted the `FitLinesPlotGrid` output plot for improved readability and formatting
- Updated the export models window and GUI output

### Theming
- Added automatic system theme syncing — the GUI now detects and updates when the macOS system theme changes
- Removed old theme files and updated `DefaultUserSettings` to use auto theming by default
- Added a new experimental `DisplayConfig` file for plot styling configuration
- Updated theme files to use a wider model line width
- GUI theming adjustments to use `ttk` instead of `tk` in several places for more consistent styling
- Updates to theming of legends and model line width/alpha
- Updates to plotting modules to work with the theming functionality

### New Features

#### Keyboard Shortcuts
- Added `V` key to toggle the visibility of the active molecule
- Added `Shift+V` to toggle all molecule visibilities at once

#### Batch Fitting
- First version of new batch fitting system using a configuration file
- Batch fitting parallelization and debugging improvements
- PDF saving improvements in `BatchFittingService`

#### Data Processing
- Refactored the `DataProcessing` module for more modular and independent operation of the fitting engine and line analyzer, removing direct dependencies on the main iSLAT instance
- Centralized stellar RV application logic to `MoleculeDict.apply_stellar_rv`, ensuring consistent RV correction across the app
- The match spectral sampling button now accounts for both stellar and per-molecule RV when matching, and also matches molecule models in addition to the sum
- Updated the HITRAN download query to use the min and max wavelength settings from user settings
- Added min/max wavelength settings to default and user settings JSON files
- Updated user settings version number to 1.2

#### LineListMaker & Data Export
- Added new `LineListMaker` module and an example notebook
- Updated `LineListMaker` and `MoleculeLineList` modules to accept a DataFrame to override partition function information
- Created two new example notebooks for generating line lists in CSV and Parquet formats
- Made modifications to `MoleculeLineList`, `FileHandling`, and `Hitran_data` modules to support new export functionalities
- Added `get_wavelengths` method to `MoleculeLineList` that returns an array of line wavelengths

#### GUI Improvements
- Updated `ControlPanel` and `FileInteractionPane` formatting
- Adjustments to control panel molecule label formatting
- Windows control panel formatting update
- Made the X buttons use the standard "x" character instead of the Unicode multiplication sign
- Updated the atomic lines load message to use the debug logger module

#### Example Notebooks
- Added `Advanced_Plotting_API.ipynb` — advanced plotting API examples
- Added `Generate_CSV_LineList_Example.ipynb` and `Generate_Parquet_Example.ipynb` — line list export examples
- Added `LineListMaker_Examples.ipynb` — LineListMaker module examples
- Updated and improved existing plotting example notebooks

#### PerformanceLogger
- Experimental update to the `PerformanceLogger` API to enable memory usage tracking

### Bug Fixes
- Fixed the toggle legend button not preserving the on/off state when plots are updated
- Fixed the toggle legend behavior to support the same state persistence as other toggles
- Fixed the atomic/saved lines toggle state not being respected when the plot view was switched
- Fixed overplotting of sample spectra when in the `FullSpectrumPlot` view
- Fixed fits not rendering correctly in the line inspection plot
- Fixed the load spectrum button not opening the file dialog when a default spectra file was set in user settings
- Fixed the output of the full spectrum to PDF
- Fixed the fit saved lines function not using the output file from the GUI correctly
- Fixed line list selection dialogs from fitting functions not correctly updating the GUI with the newly loaded line list
- Fixed line save outputting incorrect fields

### Code Quality
- Removed unused files and cleaned up old modules
- Updated README for version documentation
- Bumped version to `v5.02.03`

---

## v5.02.02 - February 08, 2026

### iSLAT Package
- Added new project toml file to define the package
- As of commit [10eec...](https://github.com/John-M87/iSLATTests/commit/10eec425a25fd7973497b52d9acfad376b0c355c) (Made iSLAT installable as a package) you will need to do one of the follow to run the notebooks on this branch:
    - Run ``pip install -e .`` from the project root (i.e. the directory where the `pyproject.toml` is located)
    - Run ``pip install -e git://github.com/John-M87/iSLATTests.git@{ tag name }#egg={ desired egg name }`` (Untested)
    - In the future you will be able to pip install as usual
- Updated some files (such as iSLAT-launch.py) to remove code that is no longer needed
- Added a new `iSLAT` console command when installed as a package
    - When you type `iSLAT` into the command line it will auto run a start script and launch the GUI
    - Could be used with a bat file if desired

### New Features

#### GUI Improvements
- Added a new plus button next to the load spectrum button
    - Lets you add multiple spectra to be used as a sample
    - When a sample is loaded the Fit Saved Lines to Sample function automatically uses it
    - When a sample is loaded you are also able to use the left and right arrow keys to dynamically move between the loaded spectra

#### Example Notebooks
- Edited all Notebooks to use a regular import for iSLAT instead of the sys path appending method

### Bug Fixes
- Updated HITRANdata file path in the FileHandling module

### Code Quality
- Bumped version to `v5.02.02`

---

## v5.02.01 - February 07, 2026

### New Features

#### User Settings System
- Added a versioning system for user settings (`user_settings_version` field in `DefaultUserSettings.json`)
- User settings are now automatically migrated when the default version is bumped: new fields are added, removed fields are pruned, and key order matches the defaults file
- If `UserSettings.json` does not exist, it is now created from `DefaultUserSettings.json` with `first_startup` set to `true`
- The `first_startup` and `reload_default_files` flags are now correctly persisted back to `false` after the first launch
- Added `save_user_settings()` function for writing settings back to disk while handling theme dict-to-string conversion and stripping runtime-only keys

#### Full Spectrum Plot
- The full spectrum plot now updates automatically when a new spectrum file is loaded
- Wavelength range is recalculated and the figure is recreated when loading data that spans a different range
- The Tk canvas is properly rebuilt when the underlying matplotlib figure changes

#### Plotting Module Refactor
- Added new modular plot classes: `BasePlot`, `FullSpectrumPlot`, `LineInspectionPlot`, `PopulationDiagramPlot`, `MainPlotGrid`
- Refactored `PlotRenderer` and `MainPlot` to use the new plot classes for improved modularity
- Plot classes now clear previous axes with `fig.clf()` so regeneration doesn't stack on top
- Line inspection plot now includes information about lines in the current selection, displayed in the data field

#### GUI Improvements
- Added a toggle for the visibility of all molecules at once by pressing the "On" label in the control panel
- Changed the "Toggle Summed Flux" button label to "Toggle Total Model"
- Repositioned the X buttons for removing input and output files
- Updated tooltips for several control panel items
- First draft of model subtraction from data
- The toggle total model button now toggles the visibility of the total model line in the main plot, and the total model is now included in the full spectrum plot

#### Full Spectrum Mode
- Reworked full spectrum mode to work with instruments other than MIRI (MIRI output remains the same)
- First draft of updated pixel sampling using [spectres](https://github.com/ACCarnall/spectres) adaptation

#### Example Notebooks
- Reorganized example notebooks into `DataTypes`, `Interactive Widgets`, `Plotting`, and `Fitting` folders
- Added `DataTypes_Overview.ipynb` — overview of the data type classes
- Added `Molecule_Examples.ipynb` — examples for the Molecule class
- Added `MoleculeDict_Examples.ipynb` — creating, managing, and visualizing molecular data
- Added `MoleculeLineList_Examples.ipynb` — working with HITRAN line list data
- Added `Plotting_Examples.ipynb` — static plotting examples
- Added `Animated_Parameter_Evolution.ipynb` — animated parameter sweep plots
- Added 10 interactive widget notebooks: Bulk Parameter Explorer, Intensity Parameter Explorer, Full Spectrum Plot, Line Inspection, Molecule Parameters, Low-Level Spectrum Viewer, Molecule Selector Dropdown, Side-by-Side Comparison, Spectrum Builder, and Visibility Toggles

### Bug Fixes
- Fixed `reload_default_files` defaulting to `True` in `check_HITRAN()`, which caused default files to reload on every launch
- Fixed molecule color access when setting molecule from a new load
- Fixed incorrect caching behavior in FileHandling `__init__.py`
- Removed unused import of `defaultdict` from `Molecule.py`

### Code Quality
- Improved type hints in `Molecule` and `MoleculeDict` classes
- Updated `bulk_update_parameters` in `MoleculeDict` to accept multiple kinds of input for specifying which molecules to update
- Updated `safe_float` method in `MoleculeDict` with improved validation
- Added `UserSettings.json` to `.gitignore` and removed it from Git tracking
- Bumped version to `v5.02.01`

---

## v5.02.00 - February 05, 2026

### New Features

#### Keyboard Shortcuts
- Added `Ctrl+S` and `Ctrl+L` shortcuts to save and load spectrum parameters from the top bar menu
- Added `Ctrl+Shift+F` shortcut for Output Full Spectrum
- Added `F` key to toggle full spectrum view
- Added `Ctrl+F` (or `Cmd+F` on Mac) to open the full spectrum in a new window
- Added up/down arrow keys to cycle through molecules in the molecule list

#### Full Spectrum View
- Added the ability to span select on the full spectrum plot, which triggers a switch back to the 3 plot grid
- Made atomic and saved lines toggles work correctly in full spectrum plot mode

#### User Settings
- Added `default_spectra_file_to_load` setting to allow loading a default spectrum file on startup
- Added `save_fit_plot_grid_directly_to_PDF` user setting
- Added setting to update save file names in the user molecules CSV
- Added new default setting for line list file in configuration files
- Added global setting for intensity calculation method (`curve_growth`, `curve_growth_no_overlap`, or `radex`)

#### Batch Fitting
- Implemented multi-processing in batch fitting service using ProcessPoolExecutor for improved performance
- Created new data processing service files for batch fitting, deblending, and line saving
- Reworked TopBar file to use new helper service classes

#### UI Improvements
- Added tooltip to the load spectrum button indicating Ctrl/Cmd click functionality
- Ctrl+clicking the load spectrum button now also loads the spectrum's parameters if they exist
- Added scroll bar to fit lines sample window
- Added minimum size for subplot cells in plot grid window
- When fitting a sample with no input line list, the selection dialog now opens automatically

### Performance Improvements
- Created new PerformanceLogger module to track execution time of key functions
- Integrated PerformanceLogger into iSlat, MoleculeDict, Molecule, Intensity, and MoleculeLineList classes
- Made GUI loading asynchronous on startup
- Added parallelization to spectrum calculations with improved performance logging
- Added caching for HITRAN data files to reduce load times
- Reworked HITRAN caching system
- Optimized the `_get_pandas_dataframe` method of the MoleculeLineList class
- Made optimization changes to the population diagram and line inspection plots
- Population diagram now renders on startup
- Fixed issue where all molecules were being rendered even when not visible

### Bug Fixes
- Fixed minor bug with setting output line measurements
- Fixed batch fitting to output correct columns
- Fixed setting span extent on main plot
- Corrected datatype handling in iSLATFileHandling.py
- Fixed minor bug when exiting the line list selection dialog
- Fixed thermal broadening calculation to properly access molar mass from MoleculeLineList
- Fixed save figure button in plot grid window
- Fixed file path for the example data folder to use the correct variable name

### Data & Configuration Updates
- Updated HITRAN catalog year from 2020 to 2024 in multiple files
- Updated default configuration and molecule list files to remove "2020" from HITRAN data file paths
- Updated default user settings file
- Reformatted output of fit saved lines CSV file to move info fields to after xmin/xmax and before flux/err fields

### Code Quality
- Changed usage of `np.trapz` to `np.trapezoid` for numerical integration (numpy deprecation fix)
- Removed debug print statements from InteractionHandler and other modules
- Cleaned up leftover code
- Updated gitignore to ignore files in the example data folder that are not the provided data files
- Updated the requirements file to specify the minimum version for several packages
- Added a version field to the user settings file, planning to implement version checking and add the file to the gitignore

---

## v5.01.00 - Beta Release

### Key Updates from v5.00.00
- Further improvements in speed and computational efficiency
- The slab model now accounts for line opacity overlap and mutual saturation, which is particularly important in the case of water ortho-para lines that overlap (see Figure 5 in [Banzatti et al. 2025](https://ui.adsabs.harvard.edu/abs/2025AJ....169..165B/abstract))
- New functionality to display and save the full spectrum to obtain something similar to Figures 1-4 in [Banzatti et al. 2025](https://ui.adsabs.harvard.edu/abs/2025AJ....169..165B/abstract) (warning: for now this is optimized for JWST MIRI spectra only!)
    - "Output full spectrum" in the spectrum parameters menu saves the full spectral range as a single-page PDF file that can be directly included in a paper
    - "Display full spectrum" in the spectrum parameters menu opens a new interactive window to view the full spectrum at the same time as the set of plots in the GUI, to display, for instance, on a second large screen
    - "Toggle full spectrum" in the main GUI toggles an interactive view of the full spectrum in the main window, by replacing the default zoomed-in plots
- The thermal broadening is calculated and displayed for reference for each molecule at the temperature set by the user
- The molar mass of each molecule is now saved in their respective HITRAN files

---

## v5.00.00 - Beta Release

### Main Updates from Version 4
- Substantial speed improvement from global refactoring of the code and data architecture
- Improved management of molecular models (now including the option to have individual RV shift and broadening)
- New function to measure line fluxes on a sample rather than on single spectrum ("Fit Saved Lines to sample")
- Several improvements to GUI organization and interactivity