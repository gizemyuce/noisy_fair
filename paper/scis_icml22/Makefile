TARGET=avg_margin
.PHONY: avg_margin clean
main:
	TEXINPUTS=.:./lib//: BSTINPUTS=.: BIBINPUTS=.: latexmk -pdf -file-line-error -synctex=1 ${TARGET}
clean:
	latexmk -C ${TARGET}
	rm -f $(TARGET).{bbl,blg,out,fls,thm,spl}
# vi:set noexpandtab:
