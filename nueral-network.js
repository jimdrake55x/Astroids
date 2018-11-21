"use strict";

/*******************
 * MATRIX FUNCTIONS
 ********************/

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