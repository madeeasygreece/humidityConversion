# humidityConversion
This module implements the conversion of the relevant humidity in the atmosphere at a given temperature, to a new expected relevant humidity at a new hypothetical temperature provided as input.

This functionality is useful because most humidity sensors in the market measure relevant humidity (that is inherently biased by the current temperature), rather than absolute humidity. The module implemented here provides the opportunity to estimate how the humidity you measure at the current temperature, would feel at another temperature.


The way the relevant humidity, of given air quality, changes at different temperatures is non-linear and you need quite some complementary information to calculate it analytically. The current module solves this problem using data collection and machine learning.

I particular, using two similar sensors measuring in parallel, we have collected XXX samples of (temperature, relative humidity) pairs of the very same air in the atmosphere before and after changing the temperature of the air with the use of a heating chamber.


Overall, each data sample consists of eight values:

[Temp1,Hum1,Temp2,Temp1,Hum1,Temp2,Hum2,Difference]

The first three values, i.e. Temp1, Hum1, Temp2, are used as the inputs for the Neural Network we train. These are the pair Temp1-Hum1 with the temperature and the humidity we have measured and the Temp2 is the temperature in which we would like to know how the atmosphere humidity would be measured.


The following five values are used as the outputs of the Neural Network. These are the two pairs Temp1-Hum1 (temperature and humidity of the atmosphere at Temp1) and Temp2-Hum2 (temperature and humidity of the same atmosphere at Temp2), plus the humidity Difference (i.e. Hum1-Hum2).

We have used partially redundant information in the output because it improves the robustness of the obtained solution. You can also use as output only the Hum2 value, but it seems the obtained solution is a bit more fragile.
