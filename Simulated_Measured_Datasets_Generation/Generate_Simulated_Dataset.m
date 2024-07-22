function main()
    % Main function to generate the dataset

    % Step 1: Get RF Environment and Configuration Parameters
    parameters = getParameters();

    % Step 2: Generate the Dataset
    % Generates a dataset of noisy received power for different angles and receivers.
    [features_powers, labels] = generateDataset(parameters);

    % Step 3: Feature Extraction - Frequency
    % Extracts frequency features from the generated powers features.
    features_frequencies = getFeaturesFrequency(features_powers);

    % Step 4: Generate Example Features
    % Generates features for a specific example angle (e.g., 357 degrees).
    [features_powers_example, features_frequency_example] = generateFeaturesExample(parameters, 357);

    % Step 5: Save the Generated Dataset
    saveDataset(features_powers, labels, features_frequencies, features_powers_example, features_frequency_example);

    % Step 6: Data Visualization - Features powers vs. Angle
    % Plots the received noisy power for different angles across all receivers.
    plotFeaturesPowerVsAngle(features_powers, parameters);

    % Step 7: Data Visualization - Features frequencies vs. Angle
    % Plots the spiking frequencies for different angles across all receivers.
    plotFeaturesFrequencyVsAngle(features_frequencies, parameters);

    % Step 8: Data Visualization - Histogram of Noisy Received Power
    % Plots a histogram of received noisy power for a specific angle and receiver.
    %(e.g., angle 129 degrees and receiver 1)
    plotHistogramOfNoisyReceivedPower(features_powers, 129, 1, parameters);
end


function parameters = getParameters()
    % --- RF Environment Parameters ---
    % Frequency in Hz.
    parameters.f = 2.4e9;

    % Speed of light in m/s.
    parameters.c = 3e8;

    % Wavelength of the signal calculated using frequency and speed of light.
    parameters.lambda = parameters.c / parameters.f;

    % Transmit power in watts.
    parameters.Pt = 0.1;

    % Transmit antenna gain (dimensionless).
    parameters.Gt = 1.5;

    % Receive antenna gain (dimensionless).
    parameters.Gr = 1.5;

    % --- RF Configuration Parameters ---
    % Total number of angles
    parameters.num_angles = 360;

    % Number of antennas in the array.
    parameters.num_antennas = 4;

    % Distance from the source to the origin in meters.
    parameters.d_s = 0.5;

    % Theta_s represents the source angle in radians.
    parameters.theta_s = linspace(0, 2*pi, parameters.num_angles + 1);

    % Distance from the receiver to the origin in meters.
    parameters.d_r = 1;

    % Angular positions of receivers, evenly spaced over 360 degrees.
    parameters.theta_r = linspace(0, 2*pi, parameters.num_antennas + 1);
    parameters.theta_r = parameters.theta_r(1:end-1); % Exclude last angle (2*pi)

    % x_r and y_r positions of receivers in the array (calculated using polar coordinates).
    parameters.x_r = parameters.d_r * cos(parameters.theta_r);
    parameters.y_r = parameters.d_r * sin(parameters.theta_r);

    % --- Dataset Generation Parameters ---
    % Resolution of the output source angle in degrees.
    parameters.resolution = 10;

    % SNR (Signal-to-Noise Ratio) level in decibels.
    parameters.snr_level = 20;

    % Number of labels (possible classes: Region, Distance, Angle).
    parameters.num_labels = 2;

    % Number of regions, typically equal to the number of antennas.
    parameters.num_regions = parameters.num_antennas;

    % Range of each region in degrees, calculated based on the total number of angles and regions.
    parameters.region_range = parameters.num_angles / parameters.num_regions;

    % Number of instances for each angle.
    parameters.num_instances_per_angle = 200;

    % Number of signal samples (used for signal processing).
    parameters.nb_samples = 10000;
end


function [features_powers, labels] = generateDataset(parameters)
    features_powers = generateFeatures(parameters);
    labels = generateLabels(parameters);
end

function features_powers = generateFeatures(parameters)
    % Initialize a matrix to hold the features. 
    % Each row corresponds to a different angle instance, and each column to an antenna.
    features_powers = zeros((parameters.num_angles) * parameters.num_instances_per_angle, parameters.num_antennas);

    % Counter to keep track of the current instance.
    instance_counter = 1;

    % Loop over each angle to compute features.
    for i = 1:parameters.num_angles

        % For each angle, generate multiple instances if required.
        for n = 1:parameters.num_instances_per_angle

            % Compute the received noisy power for the current angle and antenna positions.
            receivedNoisyPower = computeReceivedNoisyPower(parameters.x_r, parameters.y_r, parameters.theta_s(i), parameters);

            % Store the computed feature vector (received power in dBm) in
            % the corresponding row.
            features_powers(instance_counter, :) = watts2dbm(abs(receivedNoisyPower));

            % Move to the next instance.
            instance_counter = instance_counter + 1;
        end
    end

    % Display the size of the features matrix for verification.
    disp(size(features));
end

function receivedNoisyPower = computeReceivedNoisyPower(x_r, y_r, theta_s, parameters)
    % Compute the received power without noise for given x, y, and theta_s.
    receivedPower = computeReceivedPower(x_r, y_r, theta_s, parameters);

    % Initialize vectors and matrices to hold signal amplitude, signal samples, and noisy signals.
    signal_amplitude = zeros(size(receivedPower));
    signal_samples = zeros(parameters.nb_samples, length(receivedPower));
    noisy_signal_awgn = zeros(parameters.nb_samples, length(receivedPower));
    receivedNoisyPower = zeros(size(receivedPower));

    % Loop through each received power value to process signals.
    for j = 1:length(receivedPower)
        % Calculate the root mean square (RMS) amplitude of the sinusoidal signal.
        signal_amplitude(j) = sqrt(2 * receivedPower(j)); 

        % Generate a sinusoidal signal with the calculated amplitude and frequency.
        signal = generateSignal(parameters.nb_samples, signal_amplitude(j), parameters.f);
        signal_samples(:,j) = signal; 

        % Add AWGN (Additive White Gaussian Noise) to the signal based on the specified SNR level.
        noisy_signal_awgn(:,j) = addNoise(signal, parameters.snr_level);

        % Compute the power of the noisy signal and store it.
        receivedNoisyPower(j) = mean(noisy_signal_awgn(:,j).^2);
    end
end


function receivedPower = computeReceivedPower(x_r, y_r, theta_s, parameters)
    % Calculate the x_s and y_s coordinates of the source based on its angle (theta_s)
    % and distance (d_s) from the origin.
    x_s = parameters.d_s * cos(theta_s);
    y_s = parameters.d_s * sin(theta_s);

    % Compute the distance (D) from each receiver to the source (Euclidean distance). The receivers are located
    % at coordinates (x_r, y_r), and the source is at (x_s, y_s).
    D = sqrt((x_r - x_s).^2 + (y_r - y_s).^2);

    % Calculate the received power at each receiver using the Friis transmission equation.
    receivedPower = parameters.Pt * parameters.Gt * parameters.Gr * (parameters.lambda^2)./ ( (4*pi)^2 * D.^2);
end

function signal = generateSignal(nb_samples, signal_amplitude, f)
    % Create a time vector 't'. The time interval between samples is inversely
    % proportional to ten times the frequency to ensure adequate sampling.
    % This creates a time range from 0 to (nb_samples-1)/(10*f).
    t = (0:1/(10*f):(nb_samples-1)/(10*f))'; 

    % Generate the sinusoidal signal. The signal is a sine wave with the
    % specified amplitude and frequency. Each element of 't' corresponds to
    % a time point in the signal.
    signal = signal_amplitude .* sin(2*pi*f*t);

    % Note: The factor of 10 in the sampling rate is chosen to ensure that
    % the signal is well-sampled (higher than the Nyquist rate).
end

function noisy_signal = addNoise(signal, snr_level)
    % This function takes an input signal and adds Additive White Gaussian Noise (AWGN)
    % to it, based on a specified Signal-to-Noise Ratio (SNR) level.
    % Use MATLAB's built-in awgn function to add noise.
    % The 'measured' option ensures that the function measures the power of the input
    % signal and adds noise based on the specified SNR level.
    noisy_signal = awgn(signal, snr_level, 'measured');
end

function labels = generateLabels(parameters)
    % Initialize the labels matrix with zeros. The number of rows is the total number of instances
    % (angles multiplied by instances per angle). The number of columns is the number of labels per instance.
    labels = zeros((parameters.num_angles) * parameters.num_instances_per_angle, parameters.num_labels);

    % Initialize a counter to keep track of the current instance.
    instance_counter = 1;

    % Loop over each angle.
    for i = 1:parameters.num_angles
       
        % For each angle, loop over the number of instances per angle.
        for n = 1:parameters.num_instances_per_angle

            % Generate the label for the current instance based on the
            % source angle and parameters and Assign the generated label to the corresponding row in the labels matrix.
            label = labelInstances(parameters.theta_s(i), parameters);
            labels(instance_counter, :) = label;

            % Increment the instance counter.
            instance_counter = instance_counter + 1;
        end
    end
end


function label = labelInstances(theta_s, parameters)
    %   label: A 1x2 vector where the first element indicates the corresponding region,
    %   and the second element indicates the sub-region (angle within the region) index within the main region.

    % Convert the angle from radians to degrees.
    theta_s_deg = rad2deg(theta_s);

    % Determine the corresponding region index for the given angle.
    % The angle is divided by the angular span of each region.
    % Region Index could be values like 0, 1, 2, 3, ...
    region_index = floor(theta_s_deg / parameters.region_range);

    % Calculate the angle's position within its corresponding region.
    angle_within_region = rem(theta_s_deg, parameters.region_range);

    % Identify the sub-region index within the corresponding region.
    % A small offset (1e-9) ensures correct categorization at boundaries.
    % Sub-region Index could be values like 10, 20, 30, ..(for 10 degree
    % resolution) or like 1, 2, 3, .. for (1 degree resolution)
    angle_within_region_index = floor((angle_within_region + 1e-9) / parameters.resolution)* parameters.resolution;

    % Adjust the region index for angles that wrap around to the starting point.
    if region_index >= parameters.num_regions
        region_index = 0;
    end

    % Construct the label vector.
    % The first element is the corresponding region index, and the second is the sub-region index.
    label = [region_index, angle_within_region_index/parameters.resolution];
    disp(label)
    % Debug: Display the recalculated angle for verification.
    angle_theta_s = region_index * parameters.region_range + angle_within_region_index;
    disp(['Recalculated angle: ', num2str(angle_theta_s)]);
end

function features_frequencies = getFeaturesFrequency(features_powers)
    % This function interpolates frequency features from given feature powers.
    % It utilizes a cadence data of power vs frequency spikes to map feature powers to corresponding frequencies.

    % Load the power-frequency spike data from the dataset.
    [Prf, fspike] = loadCadenceData();
    
    % 'spline' interpolation is used for a smooth curve fitting.
    features_frequencies = interp1(Prf, fspike, features_powers, 'spline');
end

function [Prf, fspike] = loadCadenceData()
    % This function loads the power-frequency data from a CSV file and extracts two arrays: 
    % Prf (Power in dBm) and fspike (Frequency spikes).

    % Read the data from the CSV file into a table
    data_frequency_Power = readtable('data_frequency_power.csv');

    % Extract the columns and then transpose them
    Prf = table2array(data_frequency_Power(:, 'Prf')).';    
    fspike = table2array(data_frequency_Power(:, 'fspike')).';
end


function [features_powers_example, features_frequency_example] = generateFeaturesExample(parameters, angle_example)
    % This function generates example features for a given angle.
    % Outputs:
    %   features_powers_example: Received power in dBm for the example angle.
    %   features_frequency_example: Corresponding frequency spikes (in KHz) for the example angle.
    
    % Convert the example angle from degrees to radians.
    angle_example_rad = deg2rad(angle_example);
    
    % Compute the noisy received power for the example angle.
    NoisyReceivedPowerExample = computeReceivedNoisyPower(parameters.x_r, parameters.y_r, angle_example_rad, parameters);
    
    % Convert the received power to dBm.
    NoisyReceivedPowerExampledBm = watts2dbm(abs(NoisyReceivedPowerExample));
    
    % Assign the received power in dBm as features.
    features_powers_example = NoisyReceivedPowerExampledBm;
    
    % Generate the frequency spike features corresponding to the received power.
    features_frequency_example = getFeaturesFrequency(NoisyReceivedPowerExampledBm);
end


function saveDataset(features_powers, labels, features_frequencies, features_powers_example, features_frequency_example)
    % This function saves provided datasets into a MATLAB .mat file for future use.
    save('simulated_dataset.mat', 'features_powers', 'labels', 'features_frequencies', 'features_powers_example', 'features_frequency_example');
end


function dBm = watts2dbm(watts)
    % Convert power in watts to decibel-milliwatts (dBm).
    dBm = 10 * log10(watts * 1000);
end

function watts = dbm2watts(dBm)
    % Convert power in decibel-milliwatts (dBm) to watts.
    watts = 10^((dBm - 30) / 10); 
end

function radians = deg2rad(degrees)
    % Convert angle in degrees to radians.
    radians = degrees * (pi / 180); 
end

function degrees = rad2deg(radians)
    % Convert angle in radians to degrees.
    degrees = radians * (180 / pi);
end

function plotFeaturesPowerVsAngle(features_powers, parameters)
    % This function plots the received noisy powers (features_powers) against source angles for each receiver.

    % Reshape the feature powers for easy manipulation.
    reshaped_features_powers = reshape(features_powers, [parameters.num_instances_per_angle, parameters.num_angles, parameters.num_antennas]);

    % Calculate the mean and standard deviation of received noisy powers for each
    % angle and antenna
    mean_received_noisy_powers = squeeze(mean(reshaped_features_powers, 1));
    std_received_noisy_powers = squeeze(std(reshaped_features_powers, [], 1));

    % Create a figure with specific properties for plotting.
    fig1 = figure('Color',[1 1 1]);
    axes1 = axes('Parent',fig1, 'FontSize',25, 'FontName','Times');
    box(axes1, 'on');
    grid(axes1, 'on');
    hold(axes1, 'all');

    % Plot the mean received power with error bars representing the standard deviation.
    hold on;
    grid on;
    for i = 1:parameters.num_antennas
        errorbar(rad2deg(parameters.theta_s(1:end-1)), mean_received_noisy_powers(:,i), std_received_noisy_powers(:,i), 'LineWidth', 2, 'CapSize', 10);
    end

    % Creating an array of legend entries
    legendEntries = arrayfun(@(x) ['Receiver ' num2str(x)], 1:parameters.num_antennas, 'UniformOutput', false);
    legend(legendEntries, 'Location', 'best');

    % Label the axes and set additional plot properties.
    xlabel('Source Angle [degrees]');
    ylabel('Received Power [dBm]');
    grid on;
    hold off;
end

function plotFeaturesFrequencyVsAngle(features_frequencies, parameters)
    % This function plots the spiking frequencies against source angles for each receiver.

    % Reshape the feature powers for easy manipulation.
    reshaped_features_frequencies = reshape(features_frequencies, [parameters.num_instances_per_angle, parameters.num_angles, parameters.num_antennas]);

    % Calculate the mean and standard deviation of spiking frequencies for each
    % angle and antenna
    mean_noisy_spiking_frequencies = squeeze(mean(reshaped_features_frequencies, 1));
    std_noisy_spiking_frequencies = squeeze(std(reshaped_features_frequencies, [], 1));

    % Create a figure with specific properties for plotting.
    fig1 = figure('Color',[1 1 1]);
    axes1 = axes('Parent',fig1, 'FontSize',25, 'FontName','Times');
    box(axes1, 'on');
    grid(axes1, 'on');
    hold(axes1, 'all');

    % Plot the mean spiking frequency with error bars representing the standard deviation.
    hold on;
    grid on;
    for i = 1:parameters.num_antennas
        errorbar(rad2deg(parameters.theta_s(1:end-1)), mean_noisy_spiking_frequencies(:,i), std_noisy_spiking_frequencies(:,i), 'LineWidth', 2, 'CapSize', 10);
    end

    % Creating an array of legend entries
    legendEntries = arrayfun(@(x) ['Receiver ' num2str(x)], 1:parameters.num_antennas, 'UniformOutput', false);
    legend(legendEntries, 'Location', 'best');

    % Label the axes and set additional plot properties.
    xlabel('Source Angle [degrees]');
    ylabel('Spiking Frequencies [dBm]');
    grid on;
    hold off;
end

function plotHistogramOfNoisyReceivedPower(features_powers, angle_index, receiver_index, parameters)
    % This function plots a histogram of noisy received power values for a specific angle and receiver.
    % It helps in visualizing the distribution of received power values under noisy conditions at a particular angle.

    % Reshape the input features matrix to access specific angle and receiver data.
    reshaped_features = reshape(features_powers, [parameters.num_instances_per_angle, parameters.num_angles, parameters.num_antennas]);

    % Extract noisy received power values for the specified angle and receiver.
    noisy_received_powers = reshaped_features(:, angle_index, receiver_index);

    % Create a figure for the histogram.
    fig2 = figure('Color',[1 1 1]);
    axes2 = axes('Parent',fig2,...
    'YMinorTick','on',...
    'XMinorTick','on',...
    'XMinorGrid','on',...
    'FontSize',25,'FontName','Times');
    box(axes2, 'on');
    grid(axes2, 'on');
    hold(axes2, 'all');

    % Plot the histogram of noisy received power values.
    histogram(noisy_received_powers);
    xlabel('Received Noisy Power (dBm)');
    ylabel('Count');

    % Title the histogram with angle in degrees and receiver index.
    title(['Histogram of Received Noisy Power at ' num2str(angle_index) ' Degrees for Receiver ' num2str(receiver_index)]);
    hold off;
end
