TARGET = mmult

# CUDA
NVCC = `which nvcc`
ARCH=35

all: clean $(TARGET) 

$(TARGET): 
	$(NVCC) -o mmult -Xptxas -v -arch=sm_$(ARCH) mmult.cu 

test:
	./mmult 

golden:
	./mmult >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f *.o *~ mmult

clobber: clean
	rm -f golden* *_regcount.p stdout* stderr*
