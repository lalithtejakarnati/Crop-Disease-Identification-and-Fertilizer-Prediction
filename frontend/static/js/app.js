// ============================================================
// CROP AI ASSISTANT — FRONTEND JAVASCRIPT
// ============================================================

// ============================================================
// ELEMENTS
// ============================================================

const leafImageInput =
    document.getElementById("leafImage");

const uploadArea =
    document.getElementById("uploadArea");

const chooseImageButton =
    document.getElementById("chooseImageButton");

const imagePreviewContainer =
    document.getElementById("imagePreviewContainer");

const imagePreview =
    document.getElementById("imagePreview");

const removeImageButton =
    document.getElementById("removeImageButton");

const predictDiseaseButton =
    document.getElementById("predictDiseaseButton");

const diseasePlaceholder =
    document.getElementById("diseasePlaceholder");

const diseaseLoading =
    document.getElementById("diseaseLoading");

const diseaseResult =
    document.getElementById("diseaseResult");

const diseaseError =
    document.getElementById("diseaseError");

const predictedDisease =
    document.getElementById("predictedDisease");

const confidenceValue =
    document.getElementById("confidenceValue");

const confidenceProgress =
    document.getElementById("confidenceProgress");

const diseaseInfo =
    document.getElementById("diseaseInfo");

const diseaseDescription =
    document.getElementById("diseaseDescription");

const diseaseSymptoms =
    document.getElementById("diseaseSymptoms");

const diseaseTreatment =
    document.getElementById("diseaseTreatment");

const diseasePrevention =
    document.getElementById("diseasePrevention");

const fertilizerForm =
    document.getElementById("fertilizerForm");

const fertilizerButton =
    document.getElementById("fertilizerButton");

const fertilizerLoading =
    document.getElementById("fertilizerLoading");

const fertilizerResult =
    document.getElementById("fertilizerResult");

const fertilizerName =
    document.getElementById("fertilizerName");

const fertilizerNpk =
    document.getElementById("fertilizerNpk");

const fertilizerPurpose =
    document.getElementById("fertilizerPurpose");

const fertilizerError =
    document.getElementById("fertilizerError");


// ============================================================
// IMAGE UPLOAD
// ============================================================

leafImageInput.addEventListener(
    "change",
    function () {
        const file = this.files[0];

        if (!file) {
            return;
        }

        handleSelectedImage(file);
    }
);


// Allow clicking anywhere on upload area.

chooseImageButton.addEventListener(
    "click",
    function (event) {
        event.stopPropagation();
        leafImageInput.click();
    }
);


uploadArea.addEventListener(
    "click",
    function () {
        leafImageInput.click();
    }
);


// ============================================================
// DRAG AND DROP
// ============================================================

uploadArea.addEventListener(
    "dragover",
    function (event) {
        event.preventDefault();

        uploadArea.classList.add(
            "dragging"
        );
    }
);


uploadArea.addEventListener(
    "dragleave",
    function () {
        uploadArea.classList.remove(
            "dragging"
        );
    }
);


uploadArea.addEventListener(
    "drop",
    function (event) {
        event.preventDefault();

        uploadArea.classList.remove(
            "dragging"
        );

        const file =
            event.dataTransfer.files[0];

        if (!file) {
            return;
        }

        if (
            !file.type.startsWith(
                "image/"
            )
        ) {
            showDiseaseError(
                "Please select a valid image file."
            );

            return;
        }

        leafImageInput.files =
            event.dataTransfer.files;

        handleSelectedImage(file);
    }
);


// ============================================================
// HANDLE SELECTED IMAGE
// ============================================================

function handleSelectedImage(file) {
    hideDiseaseMessages();

    const allowedTypes = [
        "image/jpeg",
        "image/png",
    ];

    if (
        !allowedTypes.includes(
            file.type
        )
    ) {
        showDiseaseError(
            "Only JPG, JPEG and PNG images are supported."
        );

        leafImageInput.value = "";

        return;
    }

    const maxSize =
        10 * 1024 * 1024;

    if (file.size > maxSize) {
        showDiseaseError(
            "Image size must be less than 10 MB."
        );

        leafImageInput.value = "";

        return;
    }

    const reader =
        new FileReader();

    reader.onload = function (event) {
        imagePreview.src =
            event.target.result;

        imagePreviewContainer.classList.remove(
            "hidden"
        );

        uploadArea.classList.add(
            "hidden"
        );

        predictDiseaseButton.disabled =
            false;

        predictDiseaseButton.classList.remove(
            "hidden"
        );

        resetDiseaseResult();

        resetFertilizerResult();
    };

    reader.readAsDataURL(file);
}


// ============================================================
// REMOVE IMAGE
// ============================================================

removeImageButton.addEventListener(
    "click",
    function () {
        leafImageInput.value = "";

        imagePreview.src = "";

        imagePreviewContainer.classList.add(
            "hidden"
        );

        uploadArea.classList.remove(
            "hidden"
        );

        predictDiseaseButton.disabled =
            true;

        predictDiseaseButton.classList.add(
            "hidden"
        );

        resetDiseaseResult();

        resetFertilizerResult();

        hideDiseaseMessages();
    }
);


// ============================================================
// COMPLETE DISEASE + FERTILIZER PREDICTION
// ============================================================

predictDiseaseButton.addEventListener(
    "click",
    async function () {
        const file =
            leafImageInput.files[0];

        if (!file) {
            showDiseaseError(
                "Please select a leaf image first."
            );

            return;
        }

        const formData =
            new FormData();

        formData.append(
            "image",
            file
        );

        setDiseaseLoading(
            true
        );

        setFertilizerLoading(
            true
        );

        hideDiseaseMessages();
        hideFertilizerMessages();

        fertilizerResult.classList.add(
            "hidden"
        );

        try {
            const response =
                await fetch(
                    "/api/predict-disease-and-fertilizer",
                    {
                        method: "POST",
                        body: formData,
                    }
                );

            const data =
                await response.json();

            if (
                !response.ok ||
                !data.success
            ) {
                throw new Error(
                    data.error ||
                    "Disease and fertilizer prediction failed."
                );
            }

            displayDiseaseResult(
                data.disease
            );

            displayDiseaseAwareFertilizer(
                data.fertilizer
            );

        } catch (error) {
            showDiseaseError(
                error.message ||
                "Unable to complete the AI prediction."
            );

        } finally {
            setDiseaseLoading(
                false
            );

            setFertilizerLoading(
                false
            );
        }
    }
);


// ============================================================
// COLLECT FERTILIZER INPUTS
// ============================================================

function appendFertilizerInputs(
    formData
) {
    formData.append(
        "temperature",
        document.getElementById(
            "temperature"
        ).value
    );

    formData.append(
        "humidity",
        document.getElementById(
            "humidity"
        ).value
    );

    formData.append(
        "moisture",
        document.getElementById(
            "moisture"
        ).value
    );

    formData.append(
        "soil_type",
        document.getElementById(
            "soilType"
        ).value
    );
    formData.append(
        "nitrogen",
        document.getElementById(
            "nitrogen"
        ).value
    );

    formData.append(
        "potassium",
        document.getElementById(
            "potassium"
        ).value
    );

    formData.append(
        "phosphorous",
        document.getElementById(
            "phosphorous"
        ).value
    );
}


// ============================================================
// DISPLAY DISEASE RESULT
// ============================================================

function displayDiseaseResult(
    data
) {
    diseasePlaceholder.classList.add(
        "hidden"
    );

    diseaseError.classList.add(
        "hidden"
    );

    diseaseResult.classList.remove(
        "hidden"
    );

    predictedDisease.textContent =
        formatDiseaseName(
            data.prediction
        );

    const confidence =
        Number(
            data.confidence
        );

    confidenceValue.textContent =
        `${confidence.toFixed(2)}%`;

    confidenceProgress.style.width =
        `${Math.min(
            Math.max(confidence, 0),
            100
        )}%`;

    const info =
        data.disease_info || {};

    diseaseInfo.classList.remove(
        "hidden"
    );

    diseaseDescription.textContent =
        info.description ||
        "Disease information is not available.";

    diseaseTreatment.textContent =
        info.treatment ||
        "Treatment information is not available.";

    diseasePrevention.textContent =
        info.prevention ||
        "Prevention information is not available.";

    diseaseSymptoms.innerHTML =
        "";

    const symptoms =
        Array.isArray(info.symptoms)
            ? info.symptoms
            : [];

    symptoms.forEach(
        function (symptom) {
            const li =
                document.createElement(
                    "li"
                );

            li.textContent =
                symptom;

            diseaseSymptoms.appendChild(
                li
            );
        }
    );
}


// ============================================================
// DISPLAY DISEASE-AWARE FERTILIZER
// ============================================================

function displayDiseaseAwareFertilizer(
    data
) {
    fertilizerError.classList.add(
        "hidden"
    );

    fertilizerResult.classList.remove(
        "hidden"
    );

    fertilizerName.textContent =
        data.fertilizer ||
        "—";

    const fertilizerInfo =
        data.fertilizer_info || {};

    fertilizerNpk.textContent =
        data.fertilizer_npk ||
        fertilizerInfo.npk ||
        "NPK information is not available.";

    fertilizerPurpose.textContent =
        fertilizerInfo.purpose ||
        "Purpose information is not available.";

    const explanation =
        data.explanation ||
        "";

    if (explanation) {
        fertilizerPurpose.textContent =
            `${fertilizerInfo.purpose || "Recommendation generated from the fertilizer model."} ${explanation}`;
    }
}


// ============================================================
// FORMAT DISEASE NAME
// ============================================================

function formatDiseaseName(
    name
) {
    return name
        .replaceAll(
            "___",
            " — "
        )
        .replaceAll(
            "_",
            " "
        );
}


// ============================================================
// DISEASE LOADING STATE
// ============================================================

function setDiseaseLoading(
    isLoading
) {
    if (isLoading) {
        diseasePlaceholder.classList.add(
            "hidden"
        );

        diseaseResult.classList.add(
            "hidden"
        );

        diseaseError.classList.add(
            "hidden"
        );

        diseaseLoading.classList.remove(
            "hidden"
        );

        predictDiseaseButton.disabled =
            true;

        predictDiseaseButton.classList.remove(
            "hidden"
        );

        predictDiseaseButton.textContent =
            "🔄 Analyzing...";
    } else {
        diseaseLoading.classList.add(
            "hidden"
        );

        predictDiseaseButton.disabled =
            !leafImageInput.files[0];

        predictDiseaseButton.classList.remove(
            "hidden"
        );

        predictDiseaseButton.textContent =
            "🔍 Identify Disease";
    }
}


// ============================================================
// RESET DISEASE RESULT
// ============================================================

function resetDiseaseResult() {
    diseaseResult.classList.add(
        "hidden"
    );

    diseaseLoading.classList.add(
        "hidden"
    );

    diseasePlaceholder.classList.remove(
        "hidden"
    );

    predictedDisease.textContent =
        "—";

    confidenceValue.textContent =
        "0%";

    confidenceProgress.style.width =
        "0%";

    diseaseInfo.classList.add(
        "hidden"
    );

    diseaseDescription.textContent =
        "—";

    diseaseTreatment.textContent =
        "—";

    diseasePrevention.textContent =
        "—";

    diseaseSymptoms.innerHTML =
        "";
}


// ============================================================
// DISEASE ERROR
// ============================================================

function showDiseaseError(
    message
) {
    diseasePlaceholder.classList.add(
        "hidden"
    );

    diseaseResult.classList.add(
        "hidden"
    );

    diseaseLoading.classList.add(
        "hidden"
    );

    diseaseError.textContent =
        message;

    diseaseError.classList.remove(
        "hidden"
    );
}


function hideDiseaseMessages() {
    diseaseError.classList.add(
        "hidden"
    );
}


// ============================================================
// FERTILIZER FORM
// ============================================================

fertilizerForm.addEventListener(
    "submit",
    async function (event) {
        event.preventDefault();

        hideFertilizerMessages();

        fertilizerResult.classList.add(
            "hidden"
        );

        const fertilizerData = {
            temperature:
                Number(
                    document.getElementById(
                        "temperature"
                    ).value
                ),

            humidity:
                Number(
                    document.getElementById(
                        "humidity"
                    ).value
                ),

            moisture:
                Number(
                    document.getElementById(
                        "moisture"
                    ).value
                ),

            soil_type:
                document.getElementById(
                    "soilType"
                ).value,

            crop_type:
                document.getElementById(
                    "cropType"
                ).value,

            nitrogen:
                Number(
                    document.getElementById(
                        "nitrogen"
                    ).value
                ),

            potassium:
                Number(
                    document.getElementById(
                        "potassium"
                    ).value
                ),

            phosphorous:
                Number(
                    document.getElementById(
                        "phosphorous"
                    ).value
                ),
        };

        setFertilizerLoading(
            true
        );

        try {
            const response =
                await fetch(
                    "/api/predict-fertilizer",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json",
                        },

                        body:
                            JSON.stringify(
                                fertilizerData
                            ),
                    }
                );

            const data =
                await response.json();

            if (
                !response.ok ||
                !data.success
            ) {
                throw new Error(
                    data.error ||
                    "Fertilizer prediction failed."
                );
            }

            fertilizerName.textContent =
                data.fertilizer;

            const fertilizerInfo =
                data.fertilizer_info || {};

            fertilizerNpk.textContent =
                fertilizerInfo.npk ||
                "NPK information is not available.";

            fertilizerPurpose.textContent =
                fertilizerInfo.purpose ||
                "Purpose information is not available.";

            fertilizerResult.classList.remove(
                "hidden"
            );

        } catch (error) {
            showFertilizerError(
                error.message
            );

        } finally {
            setFertilizerLoading(
                false
            );
        }
    }
);


// ============================================================
// FERTILIZER LOADING STATE
// ============================================================

function setFertilizerLoading(
    isLoading
) {
    if (isLoading) {
        fertilizerLoading.classList.remove(
            "hidden"
        );

        fertilizerButton.disabled =
            true;

        fertilizerButton.textContent =
            "🔄 Analyzing...";
    } else {
        fertilizerLoading.classList.add(
            "hidden"
        );

        fertilizerButton.disabled =
            false;

        fertilizerButton.textContent =
            "🔍 Recommend Fertilizer";
    }
}


// ============================================================
// RESET FERTILIZER RESULT
// ============================================================

function resetFertilizerResult() {
    fertilizerResult.classList.add(
        "hidden"
    );

    fertilizerLoading.classList.add(
        "hidden"
    );

    fertilizerName.textContent =
        "—";

    fertilizerNpk.textContent =
        "—";

    fertilizerPurpose.textContent =
        "—";
}


// ============================================================
// FERTILIZER ERROR
// ============================================================

function showFertilizerError(
    message
) {
    fertilizerError.textContent =
        message;

    fertilizerError.classList.remove(
        "hidden"
    );
}


function hideFertilizerMessages() {
    fertilizerError.classList.add(
        "hidden"
    );
}