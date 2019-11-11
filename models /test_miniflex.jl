using Interpolations
using StaticArrays: @SVector
using Plots

# simulate some observations
t_precip = range(0, stop=900, length=3*365);
obs_precip = [t<500 ? abs(sin(t/50)*15) : 0.0  for t in t_precip]

# returns a function that interpolates the rain observations
rain = LinearInterpolation(t_precip, obs_precip, extrapolation_bc = 0)

# Our model will have 4 reservoirs but only
# the first one obtains precipitations
P_rate(t) = @SVector [rain(t), 0.0, 0.0, 0.0]

# For the similicity of the example we assume
# no potential evapotranspiration
PET_rate(t) = @SVector zeros(4)

using MiniFlex

my_model = HydroModel(
    connections = [Connection(:S1 => [:S2, :S3]),
	               Connection(:S2 => :S3),
                   Connection(:S3 => :S4)],
    P_rate = P_rate,
    PET_rate = PET_rate
)

# parameters must be a NamedTuple with the following structure
p = (θflow = ([1, 0.5],
              [5, 0.5],
              [2, 0.5],
              [1, 0.5]),
     θevap = ([1.0, 20.0],
              [2.2, 20.0],
              [3.3, 20.0],
              [2.2, 20.0]),
     θrouting = ([0.3, 0.7], # from S1 -> 30% to S2, 70% to S3
                 [1.0],      # from S2 -> 100% to S3
                 [1.0])      # from S3 -> 100% to S4
     )

	 V_init = zeros(4)
	 sol = my_model(p, V_init, 0:1000)

plot(sol, value="volume")
