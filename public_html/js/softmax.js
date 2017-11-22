
"use strict";

/*
 * Softmax classificator and Trainer
 */


function Softmax(xsize, ysize) {
    this.xsize = xsize;
    this.ysize = ysize;
    this.node = {};
    this.node["x"] = (new Float64Array(xsize)).fill(0.0);
    this.node["y"] = (new Float64Array(ysize)).fill(0.0);
    this.bias = {};
    this.bias["y"] = (new Float64Array(ysize)).fill(-0.000001);  // 0除算防止用ノイズ印加
    this.weight = {};
    this.weight["xy"] = new Array(xsize);
    for (var i = 0; i < this.weight["xy"].length; i++) {
        this.weight["xy"][i] = (new Float64Array(ysize)).fill(0.0)
        this.weight["xy"][i] = this.weight["xy"][i].map(x => { return Math.random() * 0.01 })
    }
};

Softmax.prototype._setYNode = function (yindex) {
    tyhis.ynode.fill(0.0);
    this.ynode[yindex] = 1.0;
};

Softmax.prototype.potentialFunction = function () {
    var negative_energy = 0.0;
    for (var k = 0; k < this.ysize; k++) {
        negative_energy += this.bias["y"][k] * this.node["y"][k];
        for (var i = 0; i < this.xsize; i++) {
            negative_energy += this.weight["xy"][i][k] * this.node["x"][i] * this.node["y"][k];
        }
    }

    var potential = Math.exp(negative_energy);

    return potential;
};

Softmax.prototype.potentialFunctionGivenYLabel = function (yindex) {
    var negative_energy = 0.0;
    negative_energy += this.bias["y"][yindex];

    for (var i = 0; i < this.xsize; i++) {
        negative_energy += this.weight["xy"][i][yindex] * this.node["x"][i];
    }

    var potenrial = Math.exp(negative_energy);

    return potenrial;
};


Softmax.prototype.normalizeConstant = function () {
    var value = 0.0;
    for (var k = 0; k < ysize; k++) {
        value += this.potentialFunctionGivenYLabel(k);
    }

    return value;
}

Softmax.prototype.prob = function (yindex, xdata) {
    this.node.x = xdata;
    var phi_k = this.potentialFunctionGivenYLabel(yindex);
    var z = this.normalizeConstant();

    return phi_k / z;
}

Softmax.prototype.maxY = function () {
    var prob = [];

    for (var k = 0; k < this.ysize; k++) {
        prob.push(this.prob(k));
    }

    var y = prob.indexOf(Math.max.apply(null, prob))

    return y;
};


function SoftmaxTrainer(softmax) {
    this.initGradient();

    this.optimizer = new SoftmaxOptimizer(softmax);
};

SoftmaxTrainer.prototype.initGradient = function() {
    this.gradientBias = {};
    this.gradientBias["y"] = new Float64Array(softmax.ysize);
    this.gradientWeight = {};
    this.gradientWeight["xy"] = new Array(softmax.xsize);
    for (var i = 0; i < this.gradientWeight["xy"].length; i++)
        this.gradientWeight["xy"][i] = new Float64Array(softmax.ysize);
}

SoftmaxTrainer.prototype.likelihood = function (softmax, yindex, xdata) {
    var value = softmax.prob(yindex, xdata);

    return value;
}

SoftmaxTrainer.prototype.unlikelihood = function (softmax, yindex, xdata) {
    var value = 1.0 - softmax.prob(yindex, xdata) / (softmax.ysize - 1);

    return value;
}


/*
 * @param: softmax object
 * @param: array {x:{}, y}
 * @param: learining_rate
 */
SoftmaxTrainer.prototype.train = function (softmax, dataset, learning_rate, batch_size) {
    this.optimizer.alpha = learning_rate;
    this.initGradient();


    // Online Learning(SGD)
    for (var n = 0; n < batch_size; n++) {
        this._calcTrainGradient(softmax, dataset[n]);
    }

    // update
    for (var k = 0; k < softmax.ysize; k++) {
        var gradient = this.gradientBias["y"][k] / batch_size;
        var delta = this.optimizer.deltaBias("y", k, gradient);
        var new_param = softmax.bias["y"][k] + delta;
        softmax.bias["y"][k] = new_param;
    }

    for (var i = 0; i < softmax.xsize; i++) {
        for (var k = 0; k < softmax.ysize; k++) {
            var gradient = this.gradientWeight["xy"][i][k] / batch_size;
            var delta = this.optimizer.deltaWeight("xy", i, k, gradient);
            var new_param = softmax.weight["xy"][i][k] + delta;
            softmax.weight["xy"][i][k] = new_param;
        }
    }


    // update optimizer
    this.optimizer.iteration++;
};

SoftmaxTrainer.prototype._calcTrainGradient = function(softmax, data) {
    // Gradient
    softmax.node["x"] = data["x"]
    var z = softmax.normalizeConstant();


    for (var k = 0; k < softmax.ysize; k++) {
        var grad_bias_from_data = (data.y === k) ? 1.0 : 0.0;

        var negative_energy = softmax.bias["y"][k];
        for (var i = 0; i < softmax.xsize; i++) {
            negative_energy += softmax.weight["xy"][i][k] * softmax.node.x[i];
        }
        var grad_bias_from_model = Math.exp(negative_energy) / z;


        this.gradientBias["y"][k] += grad_bias_from_data - grad_bias_from_model;

        // grad_weight
        for (var i = 0; i < softmax.xsize; i++) {
            this.gradientWeight["xy"][i][k] += data.x[i] * (grad_bias_from_data - grad_bias_from_model);
        }
    }
}

SoftmaxTrainer.prototype.forgetTrain = function (softmax, dataset, learning_rate, batch_size) {
    this.optimizer.alpha = learning_rate;
    this.initGradient();

    // Online Learning(SGD)
    for (var n = 0; n < batch_size; n++) {
        this._calcForgetTrainGradient(softmax, dataset[n]);
    }


    // update
    for (var k = 0; k < softmax.ysize; k++) {
        var gradient = this.gradientBias["y"][k] / batch_size;
        var delta = this.optimizer.deltaBias("y", k, gradient);
        var new_param = softmax.bias["y"][k] + delta;
        softmax.bias["y"][k] = new_param;
    }

    for (var i = 0; i < softmax.xsize; i++) {
        for (var k = 0; k < softmax.ysize; k++) {
            var gradient = this.gradientWeight["xy"][i][k] / batch_size;
            var delta = this.optimizer.deltaWeight("xy", i, k, gradient);
            var new_param = softmax.weight["xy"][i][k] + delta;
            softmax.weight["xy"][i][k] = new_param;
        }
    }

    // update optimizer
    this.optimizer.iteration++;
}

SoftmaxTrainer.prototype._calcForgetTrainGradient = function (softmax, data) {
    // Gradient
    softmax.node["x"] = data["x"]
    var z = softmax.normalizeConstant();
    var phi_k = softmax.potentialFunctionGivenYLabel(data.y);
    var likelihood = this.likelihood(softmax, data.y, data.x);
    var unlikelihood = this.unlikelihood(softmax, data.y, data.x);



    for (var k = 0; k < softmax.ysize; k++) {
        var grad_bias_from_data = (data.y === k) ? likelihood * z : 0.0;

        var negative_energy = softmax.bias["y"][k];
        for (var i = 0; i < softmax.xsize; i++) {
            negative_energy += softmax.weight["xy"][i][k] *  softmax.node.x[i];
        }
        var grad_bias_from_model = likelihood * Math.exp(negative_energy);

        var grad_bias = -(grad_bias_from_data - grad_bias_from_model) / (unlikelihood * z * (softmax.ysize - 1));
        this.gradientBias["y"][k] += grad_bias;

        // grad_weight
        for (var i = 0; i < softmax.xsize; i++) {
            this.gradientWeight["xy"][i][k] += data.x[i] * grad_bias;
        }
    }
}

/*
 * Optimizer: Adamax
 */
function SoftmaxOptimizer(softmax) {
    this.alpha = 0.002;
    this.beta1 = 0.9;
    this.beta2 = 0.999;
    this.epsilon = 1E-08;
    this.iteration = 1;
    this.momentBias1 = {};
    this.momentBias1["y"] = (new Float64Array(softmax.ysize)).fill(0.0);
    this.momentWeight1 = {};
    this.momentWeight1["xy"] = new Array(softmax.xsize);
    for (var i = 0; i < this.momentWeight1["xy"].length; i++)
        this.momentWeight1["xy"][i] = (new Float64Array(softmax.ysize)).fill(0.0);

    this.momentBias2 = {};
    this.momentBias2["y"] = (new Float64Array(softmax.ysize)).fill(0.0);
    this.momentWeight2 = {};
    this.momentWeight2["xy"] = new Array(softmax.xsize);
    for (var i = 0; i < this.momentWeight2["xy"].length; i++)
        this.momentWeight2["xy"][i] = (new Float64Array(softmax.ysize)).fill(0.0);

};

SoftmaxOptimizer.prototype.deltaBias = function (name, index, gradient) {
    var m = this.momentBias1[name][index] = this.beta1 * this.momentBias1[name][index] + (1.0 - this.beta1) * gradient;
    var v = this.momentBias2[name][index] = Math.max(this.beta2 * this.momentBias2[name][index], Math.abs(gradient));
    var delta = this.alpha / (1.0 - Math.pow(this.beta1, this.iteration)) * m / (v + this.epsilon);
    return delta;
};

SoftmaxOptimizer.prototype.deltaWeight = function (name, i, j, gradient) {
    var m = this.momentWeight1[name][i][j] = this.beta1 * this.momentWeight1[name][i][j] + (1.0 - this.beta1) * gradient;
    var v = this.momentWeight2[name][i][j] = Math.max(this.beta2 * this.momentWeight2[name][i][j], Math.abs(gradient));
    var delta = this.alpha / (1.0 - Math.pow(this.beta1, this.iteration)) * m / (v + this.epsilon);
    return delta;
};























/*
 * 整数(count)をN進数に変換し空配列のarrayに各桁をpushします
 */

function n_counter(count, n, array) {
    var value = count % n;
    count = Math.floor(count / n);

    array.push(value);
    if (count === 0) return;

    n_counter(count, n, array);
}


//var xsize = 10;
//var ysize = xsize;
//var train_data = [];  // 誤りデータ
//var test_data = [];  // 正解データ
//for (var n = 0; n < ysize; n++) {
//    for (var k = 0; k < ysize; k++) {
//        var data = [];
//        while (data.length < xsize) {
//            data.push(0);
//        }

//        data[n] = 1;

//        if (n === k) {  // 正解データ
//            test_data.push({ "x": data, "y": n });
//        } else {  // 誤りデータ
//            train_data.push({ "x": data, "y": k });
//        }
//    }
//}

//var softmax = new Softmax(xsize, ysize);
//var trainer = new SoftmaxTrainer(softmax);
//var softmax_t = new Softmax(xsize, ysize);
//var trainer_t = new SoftmaxTrainer(softmax);
//var alpha = 0.01;
//var iteration = 5;
//var batch_size = xsize;


//for (var i = 0; i < iteration; i++) {
//    var shuffle = function () { return Math.random() - .5 };
////    train_data.sort(shuffle);
////    test_data.sort(shuffle);

//    trainer.forgetTrain(softmax, train_data, alpha, batch_size * (batch_size - 1));
//    trainer_t.train(softmax_t, test_data, alpha, batch_size);
//}

//console.log("trained by incorrect data")
////console.log(train_data)
//for (var n = 0; n < test_data.length; n++) {
//    softmax.node["x"] = test_data[n].x;
//    var label = test_data[n].y;
//    var prob = [];
//    for (var k = 0; k < ysize; k++) {
//        prob.push(softmax.prob(k, test_data[n].x))
//    }
//    var inference = prob.indexOf(Math.max.apply(null, prob));
//    console.log("--------------------------------------------------------------------");
//    //console.log({ "input": softmax.node.x });
//    console.log({ "label": label, "inference": inference, "bool": label === inference });
//}

//console.log("")
//console.log("--------------------------------------------------------------------");
//console.log("--------------------------------------------------------------------");
//console.log("--------------------------------------------------------------------");
//console.log("")

//console.log("trained by correct data")
////console.log(test_data)
//for (var n = 0; n < test_data.length; n++) {
//    softmax_t.node["x"] = test_data[n].x;
//    label = test_data[n].y;
//    var prob = [];
//    for (var k = 0; k < ysize; k++) {
//        prob.push(softmax_t.prob(k, test_data[n].x))
//    }
//    var inference = prob.indexOf(Math.max.apply(null, prob));
//    console.log("--------------------------------------------------------------------");
//    //console.log({ "input": softmax_t.node.x });
//    console.log({ "label": label, "inference": inference, "bool": label === inference });
//}


//console.log(softmax.bias)
//console.log(softmax_t.bias)
