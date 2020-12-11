# Non-Local Image Dehazing

<img height="285px" width="200px" align="right" src="http://www.eng.tau.ac.il/~berman/images/cityscape_half.jpg">

This is the source code implementing the non-local single image dehazing algorithm described in the papers:

[Non-Local Image Dehazing. Berman, D. and Treibitz, T. and Avidan S., CVPR2016](https://www.eng.tau.ac.il/~berman/NonLocalDehazing/NonLocalDehazing_CVPR2016.pdf)

[Air-light Estimation using Haze-Lines. Berman, D. and Treibitz, T. and Avidan S., ICCP2017](https://www.eng.tau.ac.il/~berman/NonLocalDehazing/AirlightEstimation_ICCP2017.pdf)

If you use this code, please cite the appropriate paper.

#### bibtex

>    @inproceedings{NonLocalImageDehazing,    
>      title={Non-Local Image Dehazing},    
>      author={Berman, D. and Treibitz, T. and Avidan, S.},    
>      booktitle={IEEE Conf. CVPR},    
>      year={2016},
>    }
>
>    @inproceedings{AirlightEstimation,    
>      title={Air-light Estimation Using Haze-Lines},    
>      author={Berman, D. and Treibitz, T. and Avidan, S.},    
>      booktitle={IEEE Conf. ICCP},    
>      year={2017},
>    }




## Getting Started

### System Requirements:
The code requires MATLAB, and was tested on windows.

### Dehazing
1. Change MATLAB's directory to the folder the code was extracted to.
2. Run the file demo_non_local_dehazing.m in order to see the algorithm's
perfomance on the attached images.
3. in non-local-dehazing.m
	1. image_path is haze images folder.
	2. save_root is dehaze images saved folder.
### Dataset prepare
- data
	- image_path
		- 100001.jpg
		- 100002.jpg
	- save_path

### some parameters description
The gamma value is required since the image formation model applies to the linear images, where each pixel's value is directly related to the number of photons received at that location on the sensor. 
A non-linear tone-mapping is applied in camera and most of the images we see are not linear. The gamma value approximates the inverse mapping of the pixels' values to linear.

Despite that, **for new images, I recommend trying gamma=1 first** (this is the default). It often works even though it is not physically accurate.


## Authors
Dana Berman, *danamena@post.tau.ac.il*


## License
 
The software code of the non-local image dehazing algorithm is provided for non-commercial use under the attached [LICENSE.md](LICENSE.md)
