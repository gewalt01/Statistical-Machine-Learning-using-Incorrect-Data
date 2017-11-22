var xsize = 3;
var ysize = xsize;

var softmax = new Softmax(xsize, ysize);
var trainer = new SoftmaxTrainer(softmax);
make_graph(softmax);

function make_graph(softmax) {
    var xsize = softmax.xsize;
    var xnode = softmax.node.x

    //xsize = 3;
    //ysize = xsize + 1;
    //xnode = softmax.xnode;
    yvalue = softmax.maxY();



    // 出力層
    var output_layer = document.getElementById("output-layer");

    // 出力層削除
    while (0 < output_layer.children.length) {
        output_layer.children[0].remove();
    }

    // 出力層追加
    // クラスにpure-button, pure-button-primary追加せよ
    // <label><input type="radio" class="pure-button pure-button-primary train-mode" name="output" disabled /><span class="pure-button pure-button-primary">Y<sub>0</sub></span></label>
    for (var k = 0; k < ysize; k++) {
        var label = output_layer.appendChild(document.createElement("label"));
        var input = label.appendChild(document.createElement("input"))
        input.setAttribute("type", "radio");
        input.setAttribute("name", "output");
        input.disabled = true;
        input.classList.add("radio-button");
        if (k === yvalue) {
            input.checked = true;
        }

        var span = label.appendChild(document.createElement("span"));
        span.innerHTML = "Y<sub>" + k + "</sub>";
        span.classList.add("pure-button-primary");
        span.classList.add("pure-button");
    }

    // クラスにpure-button, pure-button-primary追加せよ
    // 入力層
    var input_layer = document.getElementById("input-layer");

    // 入力層削除
    while (0 < input_layer.children.length) {
        input_layer.children[0].remove();
    }

    // 入力層追加
    // クラスにpure-button, pure-button-primary追加せよ
    // <label><input type="radio" class="pure-button pure-button-primary train-mode" name="output" disabled /><span class="pure-button pure-button-primary">Y<sub>0</sub></span></label>
    for (var i = 0; i < xsize; i++) {
        var label = input_layer.appendChild(document.createElement("label"));
        var input = label.appendChild(document.createElement("input"))
        input.setAttribute("type", "checkbox");
        input.setAttribute("name", "input");
        input.classList.add("radio-button");
        if (xnode[i] == 1) {
            input.checked = true;
        }

        var span = label.appendChild(document.createElement("span"));
        span.innerHTML = "X<sub>" + i + "</sub>";
        span.classList.add("pure-button-primary");
        span.classList.add("pure-button");
    }
}

function make_data_table(dataset) {
    // var dataset = [{ "x": [0, 1, 0, 1], "y": 2 }, { "x": [1, 1, 0, 1], "y": 3 }];

    // セル全削除
    var table = document.getElementById("data-table");
    var thead = document.getElementById("data-table").children[0];
    var tbody = document.getElementById("data-table").children[1];
    thead.remove();
    tbody.remove();


    // テーブル更新
    var tr, th, td;

    // ヘッダー設定
    thead = document.createElement("thead");
    table.appendChild(thead);

    // ヘッダーY
    tr = thead.appendChild(document.createElement("tr"))
    tr.appendChild(document.createElement("th")).innerText = "Y";

    // ヘッダーX_n
    for (var i = 0; i < dataset[0].x.length; i++) {
        tr.appendChild(document.createElement("th")).innerHTML = "X<sub>" + i + "</sub>";
    }

    // ボディー設定
    tbody = table.appendChild(document.createElement("tbody"));

    // データ追加せよ
    for (var n = 0; n < dataset.length; n++) {
        var data = dataset[n];
        tr = tbody.appendChild(document.createElement("tr"));

        // ラベル項
        tr.appendChild(document.createElement("td")).innerText = data.y;

        // データ項
        for (var i = 0; i < data.x.length; i++) {
            tr.appendChild(document.createElement("td")).innerText = data.x[i];
        }
    }
}

function get_config() {
    var config = {};
    config.trainmode = 1;
    var mode_elems = document.getElementsByName("train-mode");
    for (var i = 0; i < mode_elems.length; i++) {
        if (mode_elems[i].checked) {
            config.trainmode = parseInt(mode_elems[i].value);
            break;
        }
    }
    config.xsize = parseInt(document.getElementsByName("xsize")[0].value);
    config.alpha = parseFloat(document.getElementsByName("alpha")[0].value);
    config.batchsize = parseInt(document.getElementsByName("batch-size")[0].value);
    config.iteration = parseInt(document.getElementsByName("iteration")[0].value);

    return config
}

function get_xnode() {
    var xnode = [];
    var nodes = document.getElementsByName("input");
    for (var i = 0; i < nodes.length; i++) {
        xnode.push(nodes[i].checked ? 1 : 0);
    }

    return xnode;
}

function set_ynode(label) {
    document.getElementsByName("output")[label].checked = true;
}

function on_change_xnode() {
    var xnode = get_xnode();
    softmax.node.x = xnode;
    var label = softmax.maxY();
    set_ynode(label);
}

function train() {
    var config = get_config();
    softmax = new Softmax(config.xsize, config.xsize + 1);
    trainer = new SoftmaxTrainer(softmax);

    var xsize = config.xsize;
    var ysize = config.xsize + 1;
    var batchsize = 1;
    var alpha = config.alpha;
    var iteration = config.iteration;

    var text = document.getElementById("run").innerText;
    document.getElementById("run").innerText = "学習ちゅうだ！！！";

    //データ作成
    var train_data = [];  // 誤りデータ
    var test_data = [];  // 正解データ
    for (var n = 0; n < ysize; n++) {
        for (var k = 0; k < ysize; k++) {
            var data = [];
            while (data.length < xsize) {
                data.push(0);
            }

            data[n] = 1;

            if (n === k) {  // 正解データ
                test_data.push({ "x": data, "y": n });
            } else {  // 誤りデータ
                train_data.push({ "x": data, "y": k });
            }
        }
    }

    // 表更新
    switch (config.trainmode) {
        case 0:  // train
            make_data_table(test_data);
            break;
        case 1:
            make_data_table(train_data);
            break;
    }

    // グラフ更新
    make_graph(softmax);

    // 学習
    for (var i = 0; i < iteration; i++) {
        switch (config.trainmode) {
            case 0:  // train
                batchsize = test_data.length;
                trainer.train(softmax, test_data, alpha, batchsize);
                break;
            case 1:
                batchsize = train_data.length;
                trainer.forgetTrain(softmax, train_data, alpha, batchsize);
                break;
        }
    }

    document.getElementById("run").innerText = text;


}
