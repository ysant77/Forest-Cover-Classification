/**
 * 
 * @param {FileList} files 
 * @returns 
 */
export async function fetchSegmentationResults(files) {
    const uploadData = {
        first: files.item(0)?.toString(),
        second: files.item(1),
    };

    const url = "http://localhost:8000/upload";
    const response = await fetch(url, {
        mode: "no-cors",
        method: "POST",
        headers: {
            "Content-Type": "multipart/form-data"
        },
        body: JSON.stringify(uploadData)
    });
    const data = await response.json();

    return data;
}