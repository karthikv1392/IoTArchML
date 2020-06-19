from rpy2.robjects.packages import importr

_Author_ = "Karthik Vaidhyanathan"

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects as robjects

ts=robjects.r('ts')
#import forecast package
forecast=importr('forecast')
import pandas as pd
from rpy2.robjects import pandas2ri

R_model_dict = {'ets_damped':'ets(rdata,damped=TRUE)',
                'ets':'ets(rdata)',
                'auto_arima_D_1':'auto.arima(rdata,D=1)',
                'auto_arima':'auto.arima(rdata)',
                'thetaf':'thetaf(rdata',
                'splinef':'splinef(rdata',
                'meanf':'meanf(rdata',
                'rwf':'rwf(rdata'}


def forecast_R(time_series, model=R_model_dict['thetaf'], forecast_periods=18, freq=12, confidence_level=80.0,
               return_interval=False):
    """forecasts a time series object using R models in forecast package (https://www.rdocumentation.org/packages/forecast/versions/8.1)
    Note: forecast_ts returns fitted period values as well as forecasted period values
    -Need to make sure forecast function happens in R so dask delayed works correctly


        Args:
            time_series: time series object
            model: pass in R forecast model as string that you want to evaluate, make sure you leave rdata as rdata in all calls
            forecast_periods: periods to forecast for
            freq: frequency of time series (12 is monthly)

        Returns:
             full_series: time series of fitted and forecasted values

    """
    # set frequency string to monthly start if frequency is 12
    freq_dict = {12: 'MS', 1: 'Y', 365: 'D', 4: 'QS'}
    freq_string = freq_dict[freq]

    # find the start of the time series
    start_ts = time_series[time_series.notna()].index[0]
    # find the end of the time series
    end_ts = time_series[time_series.notna()].index[-1]
    # extract actual time series
    time_series = time_series.loc[start_ts:end_ts]
    # converts to ts object in R
    time_series_R = robjects.IntVector(time_series)
    rdata = ts(time_series_R, frequency=freq)

    # if forecast model ends in f, assume its a direct forecasting object so handle it differently, no need to fit
    if model.split('(')[0][-1] == 'f':
        rstring = """
         function(rdata){
         library(forecast)
         forecasted_data<-%s,h=%s,level=c(%s))
         return(list(fitted_series=forecasted_data$fitted, predicted_series=forecasted_data$mean,lower_PI=forecasted_data$lower,upperPI=forecasted_data$upper))
         }
        """ % (model, forecast_periods, confidence_level)

    else:
        rstring = """
         function(rdata){
         library(forecast)
         fitted_model<-%s
         forecasted_data<-forecast(fitted_model,h=%s,level=c(%s))
         return(list(fitted_series=forecasted_data$fitted, predicted_series=forecasted_data$mean,lowerpi=forecasted_data$lower,upperpi=forecasted_data$upper))
         }
        """ % (model, forecast_periods, confidence_level)

    rfunc = robjects.r(rstring)
    # gets fitted and predicted series, and lower and upper prediction intervals from R model
    fitted_series, predicted_series, lowerpi, upperpi = rfunc(rdata)
    # convert IntVector representation to pandas series
    fitted_series = pd.Series(pandas2ri.ri2py(fitted_series),
                              index=pd.date_range(start=time_series[time_series.notnull()].index.min(),
                                                  periods=len(time_series[time_series.notnull()]), freq=freq_string))
    predicted_series = pandas2ri.ri2py(predicted_series)
    # get index for series
    index = pd.date_range(start=time_series.index.max(), periods=len(predicted_series) + 1, freq=freq_string)[1:]

    try:
        predicted_series = pd.Series(predicted_series, index=index)
    except:
        # need for splinef because returns array with 2 brackets
        predicted_series = pd.Series(predicted_series.ravel(), index=index)

    full_series = fitted_series.append(predicted_series)

    # if return interval set to True, then you can get the lower and upper prediction intervals back also
    if return_interval == False:
        return full_series
    return full_series, lowerpi, upperpi


full_series = forecast_R()
full_series.head()


