all: _site.yml
	Rscript -e "rmarkdown::render('index.Rmd')"
	Rscript -e "rmarkdown::render('RebalancedLeaveOneOut.Rmd')"
	Rscript -e "rmarkdown::render('RebalancedKFold.Rmd')"
	Rscript -e "rmarkdown::render('RebalancedLeavePOut.Rmd')"
	Rscript -e "rmarkdown::render('RebalancedLeaveOneOutRegression.Rmd')"
