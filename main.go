package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
)

type euclideanDistance struct {
	X2 []float64
	X1 []float64
}

func (distance *euclideanDistance) getDistance() float64 {
	var sum float64 = 0
	for i := 0; i < len(distance.X2); i++ {
		var dataPoints float64 = math.Pow((distance.X2[i] - distance.X1[i]), 2)
		sum += dataPoints
	}
	var ed float64 = math.Sqrt(sum)

	return ed
}

type KNN struct {
	k int
}

func (knn *KNN) fit(inputData []float64, X [][]float64) []float64 {
	var distances []float64
	for i := 0; i < len(X); i++ {
		d := euclideanDistance{X2: inputData, X1: X[i]}
		distances = append(distances, d.getDistance())
	}

	return distances

}

func (knn *KNN) predict(dataset [][]float64, labels []float64, distances []float64) ([][][]float64, string) {
	var completeDataset [][][]float64
	for i := 0; i < knn.k; i++ {

		var minimum float64 = math.Inf(1)
		var markedIndex int = 0
		var currentFeatures []float64
		var currentLabel float64

		for j := 0; j < len(dataset); j++ {
			if distances[j] < minimum && distances[j] != -1 {
				minimum = distances[j]
				markedIndex = j
				currentFeatures = dataset[j]
				currentLabel = labels[j]

			}
		}
		var currentArrMinimum []float64 = []float64{minimum}
		var currentArrLabel []float64 = []float64{currentLabel}
		var allIn [][]float64 = [][]float64{currentFeatures, currentArrLabel, currentArrMinimum}

		completeDataset = append(completeDataset, allIn)
		distances[markedIndex] = -1
	}

	var classZeroCount float64 = 0
	var classOneCount float64 = 0
	var classTwoCount float64 = 0
	var predictedClass string

	for i := 0; i < len(completeDataset); i++ {
		var class float64 = completeDataset[i][1][0]

		if class == 0 {
			classZeroCount++
		} else if class == 1 {
			classOneCount++
		} else {
			classTwoCount++
		}
	}

	if classZeroCount > classOneCount && classZeroCount > classTwoCount {
		predictedClass = "Iris setosa"
	} else if classOneCount > classZeroCount && classOneCount > classTwoCount {
		predictedClass = "Versi color"
	} else if classTwoCount > classZeroCount && classTwoCount > classOneCount {
		predictedClass = "Iris virginica"
	}

	return completeDataset, predictedClass
}

func loadIrisDataset() ([][]float64, []float64) {
	f, err := os.Open("iris.csv")
	if err != nil {
		log.Fatal(err)
	}

	defer f.Close()

	csvReader := csv.NewReader(bufio.NewReader(f))

	var finalDataset [][]float64

	var labels []float64

	for {
		var currentLabel float64

		rec, err := csvReader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}

		element1 := rec[0]
		element2 := rec[1]
		element3 := rec[2]
		element4 := rec[3]
		element5 := rec[4]

		feature1, error1 := strconv.ParseFloat(element1, 64)
		if error1 != nil {
			log.Fatal(error1)
		}
		feature2, error2 := strconv.ParseFloat(element2, 64)
		if error2 != nil {
			log.Fatal(error2)
		}
		feature3, error3 := strconv.ParseFloat(element3, 64)
		if error3 != nil {
			log.Fatal(error3)
		}
		feature4, error4 := strconv.ParseFloat(element4, 64)
		if error4 != nil {
			log.Fatal(error4)
		}

		var currentArray []float64 = []float64{feature1, feature2, feature3, feature4}
		finalDataset = append(finalDataset, currentArray)

		if element5 == "Iris-setosha" {
			currentLabel = 0
		} else if element5 == "Iris-versicolor" {
			currentLabel = 1
		} else if element5 == "Iris-virginica" {
			currentLabel = 2
		}

		labels = append(labels, currentLabel)
	}
	return finalDataset, labels
}

func enterFeatures() (float64, float64, float64, float64) {
	reader := bufio.NewReader(os.Stdin)
	fmt.Println("Enter sepal length in cm: ")
	sepalLength, err := reader.ReadString('\n')
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Enter sepal width in cm: ")
	sepalWidth, err := reader.ReadString('\n')
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Enter petal length in cm: ")
	petalLength, err := reader.ReadString('\n')
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Enter petal width in cm: ")
	petalWidth, err := reader.ReadString('\n')
	if err != nil {
		log.Fatal(err)
	}

	sepalLengthFloat64, err := strconv.ParseFloat(strings.TrimSuffix(sepalLength, "\n"), 64)
	if err != nil {
		log.Fatal(err)
	}
	sepalWidthFloat64, err := strconv.ParseFloat(strings.TrimSuffix(sepalWidth, "\n"), 64)
	if err != nil {
		log.Fatal(err)
	}
	petalLengthFloat64, err := strconv.ParseFloat(strings.TrimSuffix(petalLength, "\n"), 64)
	if err != nil {
		log.Fatal(err)
	}
	petalWidthFloat64, err := strconv.ParseFloat(strings.TrimSuffix(petalWidth, "\n"), 64)
	if err != nil {
		log.Fatal(err)
	}

	return sepalLengthFloat64, sepalWidthFloat64, petalLengthFloat64, petalWidthFloat64

}

func main() {
	sepalL, sepalW, petalL, petalW := enterFeatures()
	var inputFeatures = []float64{sepalL, sepalW, petalL, petalW}
	dataset, labels := loadIrisDataset()
	knn := KNN{k: 4}

	_, class := knn.predict(dataset, labels, knn.fit(inputFeatures, dataset))
	fmt.Printf("It's probably %s\n", class)
}
