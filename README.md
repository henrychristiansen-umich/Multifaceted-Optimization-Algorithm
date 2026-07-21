# Multifaceted Optimization Algorithm Abstract

Many atmospheric models are unable to produce the significant increase and subsequent recovery in the neutral density observed during geomagnetic storms. Inaccuracy of the density during these periods translates to errors in orbit propagation that reduce the accuracy of any resulting orbit predictions. With more spacecraft than ever in orbit, a significant amount of ephemeris in the form of two-line element sets (TLEs) are available. We present an improved Multifaceted Optimization Algorithm (MOA), a simple method that corrects model densities by minimizing the error between a modeled trajectory and a set of TLEs. MOA is an enhanced, restructured upgrade from the previous method. MOA utilizes ephemeris from more objects and is based on NASA’s GMAT orbit propagator. The algorithm first estimates a representative ballistic coefficient during the quiet time prior to each storm and then estimates modifications to the inputs of the NRLMSISE-00 empirical density model to minimize the difference between the modeled and TLE-derived rate of change in the semimajor axis for each spacecraft. MOA associates the corresponding adjustments to F10.7 and 3-hr a𝑝 found for each spacecraft with the TLE epoch directly instead of using linear interpolation. The median of these modifications across all spacecraft for each storm are subsequently applied to NRLMSISE-00. For validation, MOA results were systematically compared to Swarm-derived mass densities across multiple large and small geomagnetic storms that occurred during 2023 and 2024. The driver modifications produce significant improvement in the orbitaveraged neutral densities for each geomagnetic storm compared to the unchanged NRLMSISE-00 inputs.

# Setting up MOA

To setup MOA: \
(1) download the latest version of GMAT from https://sourceforge.net/projects/gmat/files/GMAT/GMAT-R2026a/ \
(2) enter the GMAT install directory and the api folder and run BuildApiStartupFile.py \
(3) download the MOA src file from GitHub
(4) create a 'data' folder in the same directoyr as teh downloaded src file
(5) Modify load_gmat.py in the src folder to include the correct filepath to the GMAT install folder (GmatInstall) \
(6) Modify run_moa.py to include the correct filepath to the output data folder (data_folder_path) and the src data folder (src_folder_path) and the GMAT install folder (gmat_path)
(7) Create an account with Space-Track.org and modify run_moa.py with username (st_usr) and password (st_pswrd)

# Running MOA
Once setup is complete, run run_moa.py to run MOA. Usage: python run_moa.py event-name start-date end-date (only 10 day windows accepted)
