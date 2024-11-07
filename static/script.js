var filename = '';
    
function changeTab(tabId) {
    const tabs = document.getElementsByClassName('tab');
    const tabHeaders = document.getElementsByClassName('tab-header');

    for (let i = 0; i < tabs.length; i++) {
        tabs[i].style.display = 'none';
        tabHeaders[i].classList.remove('active');
    }

    document.getElementById(tabId).style.display = 'block';
    document.querySelector(`[onclick="changeTab('${tabId}')"]`).classList.add('active');

    if(tabId === 'Upload' || tabId === 'Chat' || tabId === 'Explore') {
        setTimeout(fetchCollections, 100);
    }
}

function fetchCollections() {
    fetch('/list_collections')
        .then(response => {
            if (!response.ok) {
                return Promise.reject(response.statusText);
            }
            return response.json();
        })
        .then(data => {
            const collectionSelect1 = document.getElementById('collectionSelect-1');
            collectionSelect1.innerHTML = '';
            data.collections.forEach(collection => {
                const option = document.createElement('option');
                option.value = collection;
                option.textContent = collection;
                collectionSelect1.appendChild(option);
            });
            const collectionSelect2 = document.getElementById('collectionSelect-2');
            collectionSelect2.innerHTML = '';
            data.collections.forEach(collection => {
                const option = document.createElement('option');
                option.value = collection;
                option.textContent = collection;
                collectionSelect2.appendChild(option);
            });
            const collectionSelect3 = document.getElementById('collectionSelect-3');
            collectionSelect3.innerHTML = '';
            data.collections.forEach(collection => {
                const option = document.createElement('option');
                option.value = collection;
                option.textContent = collection;
                collectionSelect3.appendChild(option);
            });
        })
        .catch(error => console.log(`Fetch Error: ${error}`));
}

document.addEventListener("DOMContentLoaded", function () {
    document.getElementById('createCollectionButton').addEventListener('click', function () {
        $('#overlay').css('visibility', 'visible');
        const newCollectionName = document.getElementById('newCollectionName').value;
        fetch('/create_collection', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ name: newCollectionName }),
        }).then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                } else {
                    alert('Collection created successfully!');
                    $("#newCollectionName").val('');
                    fetchCollections();
                }
                $('#overlay').css('visibility', 'hidden');
            });
    });

    document.getElementById('deleteCollectionButton').addEventListener('click', function () {
        const collectionName = document.getElementById('collectionSelect-1').value;
        fetch('/delete_collection', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ name: collectionName }),
        }).then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                } else {
                    alert('Collection deleted successfully!');
                    fetchCollections();
                }
            });
    });

    fetch('/status').then((response) => {
        if (response.status === 200) {
            return response.json();
        } else {
            throw new Error(response.statusText);
        }
    }).then((data) => {
        document.querySelector('#database_status').innerText = data.database_status;
    }).catch((error) => {
        console.error('Error:', error);
        document.querySelector('#database_status').innerText = 'Unhealthy';
    });
});

function changeSubTab(subTabId) {
    const subTabs = document.getElementsByClassName('sub-tab');
    const subTabHeaders = document.getElementsByClassName('sub-tab-header');

    for (let i = 0; i < subTabs.length; i++) {
        subTabs[i].style.display = 'none';
        subTabHeaders[i].classList.remove('active');
    }

    document.getElementById(subTabId).style.display = 'block';
    document.querySelector(`[onclick="changeSubTab('${subTabId}')"]`).classList.add('active');
}

function validateInput(input) {
    var cleanedValue = input.value.replace(/[^0-9]/g, '');
    var intValue = parseInt(cleanedValue,10);
    input.value = isNaN(intValue) ? "" : intValue;
}

document.getElementById('fileInput').addEventListener('change', function(e) {
    var file = e.target.files[0];
    if (!file) {
        return;
    }
    filename = file.name;

    if (file.type.startsWith('application/pdf')) {
        const reader = new FileReader();
        reader.onload = function() {
            const typedarray = new Uint8Array(this.result);
            pdfjsLib.getDocument(typedarray).promise.then((pdfDoc) => {
                const numPages = pdfDoc.numPages;
                let allText = [];

                const getPageText = (pageNum) => {
                    return pdfDoc.getPage(pageNum).then((page) => {
                        return page.getTextContent().then((textContent) => {
                            const pageText = textContent.items.map(item => item.str).join(' ');
                            allText.push(pageText);
                        });
                    });
                };

                const pagePromises = [];
                for (let i = 1; i <= numPages; i++) {
                    pagePromises.push(getPageText(i));
                }

                Promise.all(pagePromises).then(() => {
                    const fileContent = allText.join('\n');
                    console.log('File content:', fileContent);
                    document.getElementById('fileContent').textContent = fileContent;

                    document.getElementById('upload').style.display = 'none';
                    document.getElementById('resetButton').style.display = 'block';
                }).catch(error => {
                    console.error("Error processing pages:", error);
                });
            }).catch(error => {
                console.error("Error getting document:", error);
            });
        };
        reader.readAsArrayBuffer(file);
    }else if(file.type.startsWith('application/vnd.openxmlformats-officedocument.wordprocessingml.document')){
        var reader = new FileReader();
        reader.onload = function(e) {
            var arrayBuffer = reader.result;

            mammoth.extractRawText({arrayBuffer: arrayBuffer})
                .then(function(result) {
                    var text = result.value; // The raw text
                    document.getElementById('fileContent').innerText = text;
                    document.getElementById('upload').style.display = 'none';
                    document.getElementById('resetButton').style.display = 'block';
                })
                .done();
        }
        reader.readAsArrayBuffer(file);
    }else {
        var reader = new FileReader();
        reader.onload = function(e) {
            var contents = e.target.result;
            document.getElementById('fileContent').textContent = contents;
            document.getElementById('upload').style.display = 'none';
            document.getElementById('resetButton').style.display = 'block';
        };
        reader.readAsText(file);
    }
}, false);

document.getElementById('resetButton').addEventListener('click', function(e) {
    document.getElementById('upload').style.display = 'block';
    document.getElementById('fileContent').textContent = '';
    this.style.display = 'none';
});

document.getElementById('newCollectionButton').addEventListener('click', function() {
    var x = document.getElementById("newCollection");
    if (x.style.display === "none") {
        x.style.display = "block";
    } else {
        x.style.display = "none";
    }
});

document.getElementById('resetNewCollectionButton').addEventListener('click', function() {
    document.getElementById('newCollectionName').value = '';
    var x = document.getElementById("newCollection");
    if (x.style.display === "none") {
        x.style.display = "block";
    } else {
        x.style.display = "none";
    }
});

document.getElementById('sendButton').addEventListener('click', function() {
    const userInput = document.getElementById('userInputField').value;
    $('#overlay').css('visibility', 'visible');

    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            message: userInput,
            collection: document.getElementById('collectionSelect-2').value,
            chunk_count: parseInt(document.getElementById('chunkCountInput').value, 10)
        })
    }).then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
            } else {
                const chunkButton = document.createElement("button");
                chunkButton.innerHTML = "Show Chunks";
                chunkButton.onclick = function () {
                    const chunkDiv = document.getElementById("chunk-data");
                    if (chunkDiv.style.display === "none") {
                        chunkDiv.style.display = "block";
                        chunkButton.innerHTML = "Hide Chunks";
                    } else {
                        chunkDiv.style.display = "none";
                        chunkButton.innerHTML = "Show Chunks";
                    }
                };

                const chunkDataElement = document.createElement("div");
                chunkDataElement.setAttribute("id", "chunk-data");
                if (data.chunks) {
                    chunkDataElement.style.display = "none";
                    chunkDataElement.innerHTML = `<code class='expandable-code'>${JSON.stringify(data.chunks, null, 2)}</code>`;

                    document.getElementById('chatHistory').innerHTML +=
                        `<p>Human: ${userInput}</p><p>AI: ${data.response}<hr />Chunks Used: ${data.chunks.length}</p>`;
                    document.getElementById('chatHistory').appendChild(chunkButton);
                    document.getElementById('chatHistory').appendChild(chunkDataElement);

                    document.getElementById('userInputField').value = '';
                    $('#overlay').css('visibility', 'hidden');
                }else{
                    chunkDataElement.style.display = "none";
                    chunkDataElement.innerHTML = `<code class='expandable-code'>${JSON.stringify([], null, 2)}</code>`;

                    document.getElementById('chatHistory').innerHTML +=
                        `<p>Human: ${userInput}</p><p>AI: ${data.response}<hr />Chunks Used: n/a</p>`;
                    document.getElementById('chatHistory').appendChild(chunkButton);
                    document.getElementById('chatHistory').appendChild(chunkDataElement);

                    document.getElementById('userInputField').value = '';
                    $('#overlay').css('visibility', 'hidden');
                }
            }
        });
});

document.getElementById('clearSessionButton').addEventListener('click', function () {
    fetch('/clear_all', { 
        method: 'POST' 
    }).then(response => response.json())
    .then(data => {
        if(data.status == 'success') {
            alert(data.message);
            window.location.reload();
        }
    });
});

let isSessionVisible = false;

document.getElementById('toggleSessionButton').addEventListener('click', function () {
    if (!isSessionVisible) {
        fetch('/show_session', { 
            method: 'GET' 
        }).then(response => response.json())
        .then(data => {
            document.getElementById('sessionData').innerHTML = JSON.stringify(data, null, 2);
            document.getElementById('toggleSessionButton').innerText = 'Hide Session';
            isSessionVisible = true;
        });
    } else {
        document.getElementById('sessionData').innerHTML = '';
        document.getElementById('toggleSessionButton').innerText = 'Show Session';
        isSessionVisible = false;
    }
});

let exploreTableInitialized = false;
let exploreTable;
document.getElementById('loadButton').addEventListener('click', function () {
    const selectedCollection = document.getElementById('collectionSelect-3').value;
    document.getElementById('overlay').style.visibility = 'visible';

    fetch('/explore?collection=' + selectedCollection)
        .then(response => response.json())
        .then(response => {
            const data = response.documents;
            console.log('response', response)
            document.getElementById('summarized-explore').innerText = response.summary;

            if (exploreTableInitialized) {
                exploreTable.clear().destroy();
                document.getElementById('exploreTable').getElementsByTagName('tbody')[0].innerHTML = '';
            }

            window.exploreTable = $('#exploreTable').DataTable({
                data: data,
                columns: [
                    { data: "source", searchable: true },
                    { data: "text", searchable: true },
                    {
                        data: null,
                        className: "center",
                        defaultContent: '<button type="button" class="edit-btn">edit</button>'
                    }
                ],
                searching: true,
                paging: true,
                ordering: true,
            });

            exploreTableInitialized = true;
            document.getElementById('overlay').style.visibility = 'hidden';
        })
        .catch(error => {
            console.log("Error: ", error);
        });
});

document.getElementById('ingestButton').addEventListener('click', function () {
    const fileContent = document.getElementById('fileContent').textContent;
    const collectionName = document.getElementById('collectionSelect-1').value;
    const chunkSize = document.getElementById('chunkSizeInput').value;
    const data = {
        text: fileContent,
        collection_name: collectionName,  
        source: filename,
        chunk_size: parseInt(chunkSize)
    };
    $('#overlay').css('visibility', 'visible');
    fetch('/ingest', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    }).then(response => response.json())
        .then(data => {
            if(data.error) {
                alert(data.error);
            } else {
                alert('Data ingested successfully');
                window.location.reload();
            }
        });
});

window.addEventListener('DOMContentLoaded', (event) => {
    document.getElementById('chunkSizeInput').addEventListener("change", validateInput);
    document.getElementById('chunkCountInput').addEventListener("change", validateInput);
});

$('#exploreTable tbody').on('click', 'button.edit-btn', function() {
    var data = window.exploreTable.row($(this).parents('tr')).data();
    console.log('rowData',data);
    document.getElementById('ogTextInput').value = data.text;

    document.getElementById('editForm').reset();

    document.getElementById('sourceInput').value = data.source;
    document.getElementById('textInput').value = data.text;

    var modal = document.getElementById("myModal");
    modal.style.display = "block";

    document.getElementsByClassName("close")[0].onclick = function() {
        modal.style.display = "none";
    }
});

function validateInput(event) {
    var numberMatch = event.target.value.match(/^-?[0-9]+/);
    var firstNumber = numberMatch ? parseInt(numberMatch[0]) : 0;

    event.target.value = firstNumber;
}

document.querySelector('.save-button').addEventListener('click', function () {
    const source = document.getElementById("sourceInput").value;
    const text = document.getElementById("textInput").value;
    const og_text = document.getElementById("ogTextInput").value;

    fetch('/update_chunk', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            action: 'save',
            collection: document.getElementById('collectionSelect-3').value,
            source: source,
            og_text: og_text,
            new_text: text
        }),
    })
    .then(response => response.json())
    .then(data => {
        alert('Chunk updated successfully');
        window.location.reload();
    });
});

document.querySelector('.delete-button').addEventListener('click', function () {
    const source = document.getElementById("sourceInput").value;
    const og_text = document.getElementById("ogTextInput").value;

    fetch('/update_chunk', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            action: 'delete',
            collection: document.getElementById('collectionSelect-3').value,
            source: source,
            og_text: og_text
        }),
    })
    .then(response => response.json())
    .then(data => {
        alert('Chunk deleted successfully');
        window.location.reload();
    });
});

document.querySelector('.cancel-button').addEventListener('click', function () {
    document.getElementById("myModal").style.display = "none";
});
