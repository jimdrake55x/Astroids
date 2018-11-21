"use strict";

/*******************
 * MATRIX FUNCTIONS
 ********************/

class NeuralNetwork{
    constructor(numInputs, numHidden, numOutputs){
        this._hidden = [];
        this._inputs = [];
        this._numInputs = numInputs;
        this._numHidden = numHidden;
        this._numOutputs = numOutputs;
        this._weights0 = new Matrix(this._numInputs, this._numHidden);
        this._weights1 = new Matrix(this._numHidden, this._numOutputs);
        this._bias0 = new Matrix(1, this._numHidden);
        this._bias1 = new Matrix(1, this._numOutputs);

        //Randomize the initial weights
        this._bias0.randomWeights();
        this._bias1.randomWeights();
        this._weights0.randomWeights();
        this._weights1.randomWeights();
    }

    get weights0(){
        return this._weights0;
    }

    set weights0(weights){
        this._weights0 = weights
    }

    get weights1(){
        return this._weights1;
    }

    set weights1(weights){
        this._weights1 = weights
    }

    get hidden(){
        return this._hidden;
    }

    set hidden(hidden){
        this._hidden = hidden;
    }

    get inputs(){
        return this._inputs;
    }

    set inputs(inputs){
        this._inputs = inputs;
    }

    get bias0(){
        return this._bias0;
    }

    set bias0(bias){
        this._bias0 = bias;
    }

    get bias1(){
        return this._bias1;
    }

    set bias1(bias){
        this._bias1 = bias;
    }

    feedForward(inputArray){
        //Convert input array to a matrix
        this.inputs = Matrix.convertFromArray(inputArray);
        
        //Find the hidden values and apply the activation function
        this.hidden = Matrix.dot(this.inputs,this.weights0);
        this.hidden = Matrix.add(this.hidden, this.bias0); // apply bias
        this.hidden = Matrix.map(this.hidden,x => sigmoid(x));

        //Find the output values and apply the activation function
        let outputs = Matrix.dot(this.hidden,this.weights1);
        outputs = Matrix.add(outputs, this.bias1); // apply bias
        outputs = Matrix.map(outputs,x => sigmoid(x));



        return outputs;

        //Apply bias??
    }

    train(inputArray, targetArray){
        //Feed the input data throught the network
        let outputs = this.feedForward(inputArray);

        //calculate the output errors(target - output);
        let targets = Matrix.convertFromArray(targetArray);
        let outputErrors = Matrix.subtract(targets,outputs);

        //Calcualte the deltas (errors * derivitive of the output)
        let outputDerivs = Matrix.map(outputs,x=>sigmoid(x,true));
        let outputDeltas = Matrix.multiply(outputErrors,outputDerivs);

        //Calculate the hidden later errors(deltas "dot" transpose of weights1)
        let weights1T = Matrix.transpose(this.weights1);
        let hiddenErrors = Matrix.dot(outputDeltas,weights1T);

        //Calculate hiden deltas(errors * derivatives of hidden)
        let hiddenDerives = Matrix.map(this.hidden, x=>sigmoid(x,true));
        let hiddenDeltas = Matrix.multiply(hiddenErrors,hiddenDerives);

        //update the weights (add transpose of layers dot deltas)
        let hiddenT = Matrix.transpose(this.hidden);
        this.weights1 = Matrix.add(this.weights1,Matrix.dot(hiddenT,outputDeltas));

        let inputsT = Matrix.transpose(this.inputs);
        this.weights0 = Matrix.add(this.weights0,Matrix.dot(inputsT,hiddenDeltas));

        this.bias1 = Matrix.add(this.bias1,outputDeltas);
        this.bias0 = Matrix.add(this.bias0,hiddenDeltas);

    }
}

function sigmoid(x, deriv = false){
    if(deriv){
        return (x*(1-x)); //Where x = sigmoid(x)
    }
    return 1 / (1 + Math.exp(-x));
}

 class Matrix{
    constructor(rows,columns, data = [])
    {
        this._rows = rows;
        this._columns = columns;
        this._data = data;

        //Initialize with zeros if no data is provided
        if(data == null || data.length == 0)
        {
            this._data = [];
            for(let i = 0; i < this._rows; i++){
                this._data[i] = [];
                for(let j = 0; j < this._columns; j++){
                    this._data[i][j] = 0;
                }
            }
        }else{
            // check data integrity
            if(data.length != rows || data[0].length != columns){
                throw new Error("Incorrect data dimensions!");
            }
            
        }
    }

    get rows(){
        return this._rows;
    }

    get columns(){
        return this._columns;
    }

    get data()
    {
        return this._data;
    }

    // apply random weights between -1 and 1
    randomWeights(){
        for(let i = 0; i < this.rows; i++){
            for(let j = 0; j < this.columns;j++){
                this.data[i][j] = Math.random() * 2 - 1;
            }
        }
    }

    //Convert array to 1 row matrix
    static convertFromArray(arr){
        return new Matrix(1,arr.length,[arr]);
    }

    //Add two matrices
    static add(m0,m1){
        Matrix.checkDimensions(m0,m1);
        let m = new Matrix(m0.rows,m0.columns);
        for(let i = 0; i < m.rows; i++){
            for(let j = 0; j < m.columns; j++){
                m.data[i][j] = m0.data[i][j] + m1.data[i][j];
            }
        }
        return m;
    }

    //Apply a function to each cell of the given matrix
    static map(m0, mFunction){
        let m = new Matrix(m0.rows,m0.columns);
        for(let i = 0; i < m.rows; i++){
            for(let j = 0; j < m.columns;j++){
                m.data[i][j] = mFunction(m0.data[i][j]);
            }
        }
        return m;
    }

    //Subtract two matricies
    static subtract(m0,m1){
        Matrix.checkDimensions(m0,m1);
        let m = new Matrix(m0.rows,m0.columns);
        for(let i = 0; i < m.rows; i++){
            for(let j = 0; j < m.columns; j++){
                m.data[i][j] = m0.data[i][j] - m1.data[i][j];
            }
        }
        return m;
    }

    //Multiply two matricies (not the dot product)
    static multiply(m0,m1){
        Matrix.checkDimensions(m0,m1);
        let m = new Matrix(m0.rows,m0.columns);
        for(let i = 0; i < m.rows; i++){
            for(let j = 0; j < m.columns; j++){
                m.data[i][j] = m0.data[i][j] * m1.data[i][j];
            }
        }
        return m;
    }

    // find the transpose of the given matrix
    static transpose(m0){
        let m = new Matrix(m0.columns,m0.rows);
        for(let i = 0; i < m0.rows; i++){
            for(let j = 0; j < m0.columns; j++){
                m.data[j][i] = m0.data[i][j];
            }
        }
        return m;
    }
    
    //Dot product of two matricies
    static dot(m0,m1){
        if(m0.columns != m1.rows){
            throw new Error("Matricies are not dot compatable");
        }
        let m = new Matrix(m0.rows,m1.columns);
        for(let i = 0; i < m.rows; i++){
            for(let j = 0; j < m.columns; j++){
                let sum = 0;
                for(let k = 0; k < m0.columns; k++){
                    sum +=  m0.data[i][k] * m1.data[k][j];
                }
                m.data[i][j] = sum;
            }
        }
        return m;
    }



    //Check Matricies have the same dimensions
    static checkDimensions(m0,m1)
    {
        if(m0.rows != m1.rows || m0.columns != m1.columns)
        {
            throw new Error("Matricies are of different dimensions");
        }
    }
}

