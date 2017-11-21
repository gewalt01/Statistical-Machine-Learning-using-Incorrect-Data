
"use strict";

/*
 * Discriminative Restricted Boltzmann Machines
 * Hidden Unit having Ising Spin: h_j \in {-1, +1}
 */
function DRBM(xsize, hsize, ysize) {
    this.xsize = xsize;
    this.hsize = hsize;
    this.ysize = ysize;
    this.node = {};
    this.node["x"] = (new Float64Array(xsize)).fill(0.0);
    this.bias = {};
    this.bias["h"] = (new Float64Array(hsize)).fill(0.0);
    this.bias["y"] = (new Float64Array(ysize)).fill(0.0);
    this.weight = {};
    this.weight["xh"] = new Array(xsize);
    for (var i = 0; i < this.weight["xh"].length; i++) {
        this.weight["xh"][i] = (new Float64Array(hsize)).fill(0.0)
        this.weight["xh"][i] = this.weight["xh"][i].map(x => { return Math.random() * 0.01 })
    }
    this.weight["hy"] = new Array(hsize);
    for (var j = 0; j < this.weight["hy"].length; j++) {
        this.weight["hy"][j] = (new Float64Array(ysize)).fill(0.0);
        this.weight["hy"][j] = this.weight["hy"][j].map(x => { return Math.random() * 0.01 })
    }
};

/*
 * Partition function
 * Worning: Partition function is very huge.
 * So it's easy to overflow.
 */
DRBM.prototype.normalizeConstant = function () {
    var value = Math.pow(2, this.hsize) * this.normalizeConstantDiv2H();
    return value;
};

/*
 * z = Z / 2^|H|
 */
DRBM.prototype.normalizeConstantDiv2H = function () {
    var value = 0.0;
    for (var k = 0; k < this.ysize; k++) {
        var k_val = Math.exp(this.bias["y"][k]);
        for (var j = 0; j < this.hsize; j++) {
            k_val *= Math.cosh(this.muJK(j, k));
        }
        value += k_val;
    }
    return value;
};

DRBM.prototype.muJK = function (hindex, yindex) {
    var value = this.bias["h"][hindex] + this.weight["hy"][hindex][yindex];
    for (var i = 0; i < this.xsize; i++) {
        value += this.weight["xh"][i][hindex] * this.node["x"][i];
    }
    return value;
};

DRBM.prototype.muJKMatrix = function () {
    var mujk = new Array(this.hsize);
    for (var j = 0; j < this.hsize; j++) {
        mujk[j] = new Float64Array(this.ysize)
        for (var k = 0; k < this.ysize; k++) {
            mujk[j][k] = this.muJK(j, k);
        }
    }
    return mujk;
};


DRBM.prototype.prob = function (yindex) {
    var z_k = this.normalizeConstantDiv2H();
    var value = this.probGivenZ(yindex, z_k);
    return value;
};

DRBM.prototype.probGivenZ = function (yindex, z) {
    var z_k = z;
    var potential = 0.0; {
        var k_val = Math.exp(this.bias["y"][yindex]);
        for (var j = 0; j < this.hsize; j++) {
            var mu_j = this.muJK(j, yindex);
            k_val *= Math.cosh(mu_j);
        }
        potential += k_val;
    }
    var value = potential / z_k;
    return value;
};

DRBM.prototype.maxY = function () {
    var z_k = this.normalizeConstantDiv2H();
    var prob = [];

    for (var k = 0; k < this.ysize; k++) {
        prob.push(this.probGivenZ(k, z_k));
    }

    var y = prob.indexOf(Math.max.apply(null, prob))

    return y;
};


DRBM.prototype.expectedValueXH = function (xindex, hindex) {
    var z = this.normalizeConstantDiv2H();
    var value = this.node["x"][xindex] * this.expectedValueHGivenZ(hindex, z);
    return value;
};

DRBM.prototype.expectedValueXHGivenZ = function (xindex, hindex, z) {
    var value = this.node["x"][xindex] * this.expectedValueHGivenZ(hindex, z);
    return value;
};

DRBM.prototype.expectedValueXHGivenZGivenMu = function (xindex, hindex, z, mujk) {
    var value = this.node["x"][xindex] * this.expectedValueHGivenZGivenMu(hindex, z, mujk);
    return value;
};


DRBM.prototype.expectedValueH = function (hindex) {
    var z = this.normalizeConstantDiv2H();
    var value = this.expectedValueHGivenZ(hindex, z);
    return value;
};

DRBM.prototype.expectedValueHGivenZ = function (hindex, z) {
    var lindex = Array.from({
        length: this.hsize
    }, (v, k) => k);
    lindex.splice(hindex, 1);
    var value = 0.0;
    for (var k = 0; k < this.ysize; k++) {
        var k_val = Math.exp(this.bias["y"][k]);
        for (var l of lindex) {
            k_val *= Math.cosh(this.muJK(l, k));
        }
        k_val *= Math.sinh(this.muJK(hindex, k));
        value += k_val;
    }
    value /= z;
    return value;
};

DRBM.prototype.expectedValueHGivenZGivenMu = function (hindex, z, mujk) {
    var lindex = Array.from({
        length: this.hsize
    }, (v, k) => k);
    lindex.splice(hindex, 1);
    var value = 0.0;
    for (var k = 0; k < this.ysize; k++) {
        var k_val = Math.exp(this.bias["y"][k]);
        for (var l of lindex) {
            k_val *= Math.cosh(mujk[l][k]);
        }
        k_val *= Math.sinh(mujk[hindex][k]);
        value += k_val;
    }
    value /= z;
    return value;
};

DRBM.prototype.expectedValueHY = function (hindex, yindex) {
    var z = this.normalizeConstantDiv2H();
    var value = this.expectedValueHYGivenZ(hindex, yindex, z);
    return value;
};

DRBM.prototype.expectedValueHYGivenZ = function (hindex, yindex, z) {
    var lindex = Array.from({
        length: this.hsize
    }, (v, k) => k);
    lindex.splice(hindex, 1);
    var value = Math.exp(this.bias["y"][yindex]);
    for (var l of lindex) {
        value *= Math.cosh(this.muJK(l, yindex));
    }
    value *= Math.sinh(this.muJK(hindex, yindex));
    value /= z;
    return value;
};

DRBM.prototype.expectedValueHYGivenZGivenMu = function (hindex, yindex, z, mujk) {
    var lindex = Array.from({
        length: this.hsize
    }, (v, k) => k);
    lindex.splice(hindex, 1);
    var value = Math.exp(this.bias["y"][yindex]);
    for (var l of lindex) {
        value *= Math.cosh(mujk[l][yindex]);
    }
    value *= Math.sinh(mujk[hindex][yindex]);
    value /= z;
    return value;
};

DRBM.prototype.expectedValueY = function (yindex) {
    var z = this.normalizeConstantDiv2H();
    var value = this.expectedValueYGivenZ(yindex, z);
    return value;
};

DRBM.prototype.expectedValueYGivenZ = function (yindex, z) {
    var lindex = Array.from({
        length: this.hsize
    }, (v, k) => k);
    var value = Math.exp(this.bias["y"][yindex]);
    for (var l of lindex) {
        value *= Math.cosh(this.muJK(l, yindex));
    }
    value /= z;
    return value;
};

DRBM.prototype.expectedValueYGivenZGivenMu = function (yindex, z, mujk) {
    var lindex = Array.from({
        length: this.hsize
    }, (v, k) => k);
    var value = Math.exp(this.bias["y"][yindex]);
    for (var l of lindex) {
        value *= Math.cosh(mujk[l][yindex]);
    }
    value /= z;
    return value;
};

function DRBMTrainer(drbm) {
    this.initGradient(drbm);
    this.optimizer = new DRBMOptimizer(drbm);
};

DRBMTrainer.prototype.initGradient = function (drbm) {
    this.gradientBias = {};
    this.gradientBias["h"] = new Float64Array(drbm.hsize);
    this.gradientBias["y"] = new Float64Array(drbm.ysize);
    this.gradientWeight = {};
    this.gradientWeight["xh"] = new Array(drbm.xsize);
    for (var i = 0; i < this.gradientWeight["xh"].length; i++)
        this.gradientWeight["xh"][i] = new Float64Array(drbm.hsize);
    this.gradientWeight["hy"] = new Array(drbm.hsize);
    for (var j = 0; j < this.gradientWeight["hy"].length; j++)
        this.gradientWeight["hy"][j] = new Float64Array(drbm.ysize);
}

/*
 * @param: drbm object
 * @param: array {x:{}, y}
 * @param: learining_rate
 */
DRBMTrainer.prototype.train = function (drbm, dataset, learning_rate, batch_size) {
    this.optimizer.alpha = learning_rate;
    this.initGradient(drbm);


    // Online Learning(SGD)
    for (var n = 0; n < batch_size; n++) {
        this._calcTrainGradient(drbm, dataset[n]);
    }



    // update
    for (var i = 0; i < drbm.xsize; i++) {
        for (var j = 0; j < drbm.hsize; j++) {
            var gradient = this.gradientWeight["xh"][i][j] / batch_size;
            var delta = this.optimizer.deltaWeight("xh", i, j, gradient);
            var new_param = drbm.weight["xh"][i][j] + delta;
            drbm.weight["xh"][i][j] = new_param;
        }
    }
    for (var j = 0; j < drbm.hsize; j++) {
        var gradient = this.gradientBias["h"][j] / batch_size;
        var delta = this.optimizer.deltaBias("h", j, gradient);
        var new_param = drbm.bias["h"][j] + delta;
        drbm.bias["h"][j] = new_param;
    }
    for (var j = 0; j < drbm.hsize; j++) {
        for (var k = 0; k < drbm.ysize; k++) {
            var gradient = this.gradientWeight["hy"][j][k] / batch_size;
            var delta = this.optimizer.deltaWeight("hy", j, k, gradient);
            var new_param = drbm.weight["hy"][j][k] + delta;
            drbm.weight["hy"][j][k] = new_param;
        }
    }
    for (var k = 0; k < drbm.ysize; k++) {
        var gradient = this.gradientBias["y"][k] / batch_size;
        var delta = this.optimizer.deltaBias("y", k, gradient);
        var new_param = drbm.bias["y"][k] + delta;
        drbm.bias["y"][k] = new_param;
    }

    // update optimizer
    this.optimizer.iteration++;
};

DRBMTrainer.prototype._calcTrainGradient = function (drbm, data) {
    // Gradient
    drbm.node["x"] = data["x"];
    var z = drbm.normalizeConstantDiv2H();
    var mujk = drbm.muJKMatrix()

    for (var i = 0; i < drbm.xsize; i++) {
        for (var j = 0; j < drbm.hsize; j++) {
            var gradient = this.dataMeanXHGivenMu(drbm, data, i, j, mujk) - drbm.expectedValueXHGivenZGivenMu(i, j, z, mujk);
            this.gradientWeight["xh"][i][j] += gradient;
        }
    }
    for (var j = 0; j < drbm.hsize; j++) {
        var gradient = this.dataMeanHGivenMu(drbm, data, j, mujk) - drbm.expectedValueHGivenZGivenMu(j, z, mujk);
        this.gradientBias["h"][j] += gradient;
    }
    for (var j = 0; j < drbm.hsize; j++) {
        for (var k = 0; k < drbm.ysize; k++) {
            var gradient = this.dataMeanHYGivenMu(drbm, data, j, k, mujk) - drbm.expectedValueHYGivenZGivenMu(j, k, z, mujk);
            this.gradientWeight["hy"][j][k] += gradient;
        }
    }
    for (var k = 0; k < drbm.ysize; k++) {
        var gradient = this.dataMeanY(drbm, data, k) - drbm.expectedValueYGivenZGivenMu(k, z, mujk);
        this.gradientBias["y"][k] += gradient;
    }
}


DRBMTrainer.prototype.forgetTrain = function (drbm, dataset, learning_rate, batch_size) {
    this.optimizer.alpha = learning_rate;
    this.initGradient(drbm);

    // Online Learning(SGD)
    for (var n = 0; n < batch_size; n++) {
        this._calcForgetTrainGradient(drbm, dataset[n]);
    }


    // update
    for (var i = 0; i < drbm.xsize; i++) {
        for (var j = 0; j < drbm.hsize; j++) {
            var gradient = this.gradientWeight["xh"][i][j] / batch_size;
            var delta = this.optimizer.deltaWeight("xh", i, j, gradient);
            var new_param = drbm.weight["xh"][i][j] + delta;
            drbm.weight["xh"][i][j] = new_param;
        }
    }
    for (var j = 0; j < drbm.hsize; j++) {
        var gradient = this.gradientBias["h"][j] / batch_size;
        var delta = this.optimizer.deltaBias("h", j, gradient);
        var new_param = drbm.bias["h"][j] + delta;
        drbm.bias["h"][j] = new_param;
    }
    for (var j = 0; j < drbm.hsize; j++) {
        for (var k = 0; k < drbm.ysize; k++) {
            var gradient = this.gradientWeight["hy"][j][k] / batch_size;
            var delta = this.optimizer.deltaWeight("hy", j, k, gradient);
            var new_param = drbm.weight["hy"][j][k] + delta;
            drbm.weight["hy"][j][k] = new_param;
        }
    }
    for (var k = 0; k < drbm.ysize; k++) {
        var gradient = this.gradientBias["y"][k] / batch_size;
        var delta = this.optimizer.deltaBias("y", k, gradient);
        var new_param = drbm.bias["y"][k] + delta;
        drbm.bias["y"][k] = new_param;
    }

    // update optimizer
    this.optimizer.iteration++;
}

DRBMTrainer.prototype._calcForgetTrainGradient = function (drbm, data) {
    drbm.node["x"] = data["x"]
    var z = drbm.normalizeConstantDiv2H();
    // Gradient
    var mujk = drbm.muJKMatrix()

    var likelihood = this.likelihoodGivenZGivenMu(drbm, data, z, mujk);
    var unlikelihood = this.unlikelihoodGivenZGivenMu(drbm, data, z, mujk);


    for (var i = 0; i < drbm.xsize; i++) {
        for (var j = 0; j < drbm.hsize; j++) {
            var gradient = data["x"][i] * likelihood * Math.tanh(mujk[j][data["y"]]) - likelihood * drbm.expectedValueXHGivenZGivenMu(i, j, z, mujk);
            gradient /= -unlikelihood * (drbm.ysize - 1);
            this.gradientWeight["xh"][i][j] += gradient;
        }
    }
    for (var j = 0; j < drbm.hsize; j++) {
        var gradient = likelihood * Math.tanh(mujk[j][data["y"]]) - likelihood * drbm.expectedValueHGivenZGivenMu(j, z, mujk);
        gradient /= -unlikelihood * (drbm.ysize - 1);
        this.gradientBias["h"][j] += gradient;
    }
    for (var j = 0; j < drbm.hsize; j++) {
        for (var k = 0; k < drbm.ysize; k++) {
            var y_k = data["y"] == k ? 1 : 0;
            var gradient = y_k * likelihood * Math.tanh(mujk[j][data["y"]]) - likelihood * drbm.expectedValueHYGivenZGivenMu(j, k, z, mujk);
            gradient /= -unlikelihood * (drbm.ysize - 1);
            this.gradientWeight["hy"][j][k] += gradient;
        }
    }
    for (var k = 0; k < drbm.ysize; k++) {
        var y_k = data["y"] == k ? 1 : 0;
        var gradient = y_k * likelihood - likelihood * drbm.expectedValueYGivenZGivenMu(k, z, mujk);
        gradient /= -unlikelihood * (drbm.ysize - 1);
        this.gradientBias["y"][k] += gradient;
    }
}

DRBMTrainer.prototype.dataMeanXH = function (drbm, data, xindex, hindex) {
    var mu = drbm.bias["h"][hindex] + drbm.weight["hy"][hindex][data.y];
    for (var i = 0; i < drbm.xsize; i++) {
        mu += drbm.weight["xh"][i][hindex] * data.x[i];
    }
    var value = data.x[xindex] * Math.tanh(mu);
    return value;
};

DRBMTrainer.prototype.dataMeanXHGivenMu = function (drbm, data, xindex, hindex, mujk) {
    var value = data.x[xindex] * Math.tanh(mujk[hindex][data.y]);
    return value;
};


DRBMTrainer.prototype.dataMeanH = function (drbm, data, hindex) {
    var mu = drbm.bias["h"][hindex] + drbm.weight["hy"][hindex][data.y];
    for (var i = 0; i < drbm.xsize; i++) {
        mu += drbm.weight["xh"][i][hindex] * data.x[i];
    }
    var value = Math.tanh(mu);
    return value;
};

DRBMTrainer.prototype.dataMeanHGivenMu = function (drbm, data, hindex, mujk) {
    var value = Math.tanh(mujk[hindex][data.y]);
    return value;
};


DRBMTrainer.prototype.dataMeanHY = function (drbm, data, hindex, yindex) {
    if (yindex !== data.y) return 0.0;
    var mu = drbm.bias["h"][hindex] + drbm.weight["hy"][hindex][data.y];
    for (var i = 0; i < drbm.xsize; i++) {
        mu += drbm.weight["xh"][i][hindex] * data.x[i];
    }
    var value = Math.tanh(mu);
    return value;
};

DRBMTrainer.prototype.dataMeanHYGivenMu = function (drbm, data, hindex, yindex, mujk) {
    if (yindex !== data.y) return 0.0;
    var value = Math.tanh(mujk[hindex][data.y]);
    return value;
};

DRBMTrainer.prototype.likelihoodGivenZGivenMu = function (drbm, data, z, mujk) {
    drbm.node["x"] = data["x"];
    var l = 1.0 / z * Math.exp(drbm.bias["y"][data.y]);

    for (var j = 0; j < drbm.hsize; j++) {
        l *= Math.cosh(mujk[j][data.y]);
    }

    return l;
}

DRBMTrainer.prototype.unlikelihoodGivenZGivenMu = function (drbm, data, z, mujk) {
    var l = 1.0 - this.likelihoodGivenZGivenMu(drbm, data, z, mujk);

    return l;
}


DRBMTrainer.prototype.dataMeanY = function (drbm, data, yindex) {
    var value = (yindex !== data.y) ? 0.0 : 1.0;
    return value;
};

/*
 * Optimizer: Adamax
 */
function DRBMOptimizer(drbm) {
    this.alpha = 0.002;
    this.beta1 = 0.9;
    this.beta2 = 0.999;
    this.epsilon = 1E-08;
    this.iteration = 1;
    this.momentBias1 = {};
    this.momentBias1["h"] = (new Float64Array(drbm.hsize)).fill(0.0);
    this.momentBias1["y"] = (new Float64Array(drbm.ysize)).fill(0.0);
    this.momentWeight1 = {};
    this.momentWeight1["xh"] = new Array(drbm.xsize);
    for (var i = 0; i < this.momentWeight1["xh"].length; i++)
        this.momentWeight1["xh"][i] = (new Float64Array(drbm.hsize)).fill(0.0);
    this.momentWeight1["hy"] = new Array(drbm.hsize);
    for (var j = 0; j < this.momentWeight1["hy"].length; j++)
        this.momentWeight1["hy"][j] = (new Float64Array(drbm.ysize)).fill(0.0);

    this.momentBias2 = {};
    this.momentBias2["h"] = (new Float64Array(drbm.hsize)).fill(0.0);
    this.momentBias2["y"] = (new Float64Array(drbm.ysize)).fill(0.0);
    this.momentWeight2 = {};
    this.momentWeight2["xh"] = new Array(drbm.xsize);
    for (var i = 0; i < this.momentWeight2["xh"].length; i++)
        this.momentWeight2["xh"][i] = (new Float64Array(drbm.hsize)).fill(0.0);
    this.momentWeight2["hy"] = new Array(drbm.hsize);
    for (var j = 0; j < this.momentWeight2["hy"].length; j++)
        this.momentWeight2["hy"][j] = (new Float64Array(drbm.ysize)).fill(0.0);
};

DRBMOptimizer.prototype.deltaBias = function (name, index, gradient) {
    var m = this.momentBias1[name][index] = this.beta1 * this.momentBias1[name][index] + (1.0 - this.beta1) * gradient;
    var v = this.momentBias2[name][index] = Math.max(this.beta2 * this.momentBias2[name][index], Math.abs(gradient));
    var delta = this.alpha / (1.0 - Math.pow(this.beta1, this.iteration)) * m / (v + this.epsilon);
    return delta;
};

DRBMOptimizer.prototype.deltaWeight = function (name, i, j, gradient) {
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


//var xsize = 4;
//var hsize = 10;
//var ysize = xsize + 1;
//var train_data = [];  // 誤りデータ
//var test_data = [];  // 正解データ
//for (var n = 0; n < Math.pow(2, xsize); n++) {
//    for (var k = 0; k < ysize; k++) {
//        var data = [];
//        n_counter(n, 2, data)
//        while (data.length < xsize) {
//            data.push(0);
//        }
//        var sum = data.reduce((a, x) => a += x, 0);

//        if (k === sum) {  // 正解データ
//            test_data.push({ "x": data, "y": k });
//        } else {  // 誤りデータ
//            train_data.push({ "x": data, "y": k });
//        }
//    }
//}


//var drbm = new DRBM(xsize, hsize, ysize);
//var trainer = new DRBMTrainer(drbm);
//var drbm_t = new DRBM(xsize, hsize, ysize);
//var trainer_t = new DRBMTrainer(drbm);
//var alpha = 0.1;
//var iteration = 500;


//for (var i = 0; i < iteration; i++) {
//    var shuffle = function () { return Math.random() - .5 };
//    //train_data.sort(shuffle);
//    //test_data.sort(shuffle);
//    var train_data_batch_size = train_data.length;
//    var test_data_batch_size = test_data.length;

//    trainer.forgetTrain(drbm, train_data, alpha, train_data_batch_size);
//    trainer_t.train(drbm_t, test_data, alpha, test_data_batch_size);
//}

//console.log("trained by incorrect data")
////console.log(train_data)
//for (var n = 0; n < test_data.length; n++) {
//    drbm.node["x"] = test_data[n].x;
//    var label = test_data[n].y;
//    var prob = [];
//    for (var k = 0; k < ysize; k++) {
//        prob.push(drbm.prob(k, test_data[n].x))
//    }
//    var inference = prob.indexOf(Math.max.apply(null, prob));
//    console.log("--------------------------------------------------------------------");
//    //console.log({ "input": drbm.node.x });
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
//    drbm_t.node["x"] = test_data[n].x;
//    label = test_data[n].y;
//    var prob = [];
//    for (var k = 0; k < ysize; k++) {
//        prob.push(drbm_t.prob(k, test_data[n].x))
//    }
//    var inference = prob.indexOf(Math.max.apply(null, prob));
//    console.log("--------------------------------------------------------------------");
//    //console.log({ "input": drbm_t.node.x });
//    console.log({ "label": label, "inference": inference, "bool": label === inference });
//}


//console.log(drbm.bias)
//console.log(drbm_t.bias)
