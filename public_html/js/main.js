var xsize = 3;
var hsize = 10;
var ysize = xsize + 1;

var drbm = new DRBM(xsize, hsize, ysize);
var trainer = new DRBMTrainer(drbm);
make_graph(drbm);

function make_graph(drbm) {
    var xsize = drbm.xsize;
    var ysize = xsize + 1;
    var xnode = drbm.node.x

    //xsize = 3;
    //ysize = xsize + 1;
    //xnode = drbm.xnode;
    yvalue = drbm.maxY();



    // �o�͑w
    var output_layer = document.getElementById("output-layer");

    // �o�͑w�폜
    while (0 < output_layer.children.length) {
        output_layer.children[0].remove();
    }

    // �o�͑w�ǉ�
    // �N���X��pure-button, pure-button-primary�ǉ�����
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

    // �N���X��pure-button, pure-button-primary�ǉ�����
    // ���͑w
    var input_layer = document.getElementById("input-layer");

    // ���͑w�폜
    while (0 < input_layer.children.length) {
        input_layer.children[0].remove();
    }

    // ���͑w�ǉ�
    // �N���X��pure-button, pure-button-primary�ǉ�����
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

    // �Z���S�폜
    var table = document.getElementById("data-table");
    var thead = document.getElementById("data-table").children[0];
    var tbody = document.getElementById("data-table").children[1];
    thead.remove();
    tbody.remove();


    // �e�[�u���X�V
    var tr, th, td;

    // �w�b�_�[�ݒ�
    thead = document.createElement("thead");
    table.appendChild(thead);

    // �w�b�_�[Y
    tr = thead.appendChild(document.createElement("tr"))
    tr.appendChild(document.createElement("th")).innerText = "Y";

    // �w�b�_�[X_n
    for (var i = 0; i < dataset[0].x.length; i++) {
        tr.appendChild(document.createElement("th")).innerHTML = "X<sub>" + i + "</sub>";
    }

    // �{�f�B�[�ݒ�
    tbody = table.appendChild(document.createElement("tbody"));

    // �f�[�^�ǉ�����
    for (var n = 0; n < dataset.length; n++) {
        var data = dataset[n];
        tr = tbody.appendChild(document.createElement("tr"));

        // ���x����
        tr.appendChild(document.createElement("td")).innerText = data.y;

        // �f�[�^��
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
    config.hsize = parseInt(document.getElementsByName("hsize")[0].value);
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
    drbm.node.x = xnode;
    var label = drbm.maxY();
    set_ynode(label);
}

function train() {
    var config = get_config();
    drbm = new DRBM(config.xsize, config.hsize, config.xsize + 1);
    trainer = new DRBMTrainer(drbm);

    var xsize = config.xsize;
    var hsize = config.hsize;
    var ysize = config.xsize + 1;
    var batchsize = 1;
    var alpha = config.alpha;
    var iteration = config.iteration;

    var text = document.getElementById("run").innerText;
    document.getElementById("run").innerText = "�w�K���イ���I�I�I";

    //�f�[�^�쐬
    var train_data = [];  // ���f�[�^
    var test_data = [];  // �����f�[�^
    for (var n = 0; n < Math.pow(2, xsize); n++) {
        for (var k = 0; k < ysize; k++) {
            var data = [];
            n_counter(n, 2, data)
            while (data.length < xsize) {
                data.push(0);
            }
            var sum = data.reduce((a, x) => a += x, 0);

            if (k === sum) {  // �����f�[�^
                test_data.push({ "x": data, "y": k });
            } else {  // ���f�[�^
                train_data.push({ "x": data, "y": k });
            }
        }
    }

    // �\�X�V
    switch (config.trainmode) {
        case 0:  // train
            make_data_table(test_data);
            break;
        case 1:
            make_data_table(train_data);
            break;
    }

    // �O���t�X�V
    make_graph(drbm);

    // �w�K
    for (var i = 0; i < iteration; i++) {
        switch (config.trainmode) {
            case 0:  // train
                batchsize = test_data.length;
                trainer.train(drbm, test_data, alpha, batchsize);
                break;
            case 1:
                batchsize = train_data.length;
                trainer.forgetTrain(drbm, train_data, alpha, batchsize);
                break;
        }
    }

    document.getElementById("run").innerText = text;


}
