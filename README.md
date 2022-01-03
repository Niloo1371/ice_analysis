# ice-data-analysis
Bragg CDI analysis of hexagonal ice, including peak finding and peak statistics

-----------------------------

Inspect ice data, check all options using the help flag `-h`. Example usage follows below.

* inspect only mode (recommended), use a mask to make bad pixels gray, set the upper intensity limit to 5000 ADU, output more information to screen:

`./inspectIce.py -r 0144 -i -m r0144_type0-pixel_mask_r0166_borders+auto+variance+gain.h5 -M 5000 -v`
