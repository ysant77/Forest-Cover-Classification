import { fail, redirect } from '@sveltejs/kit';
import { setProcessedData } from '$lib/stores'; 

export const actions = {
    default: async ({ request }) => {
        const formData = await request.formData();
        const files = formData?.getAll('many');
        // Prepare data for upload
        let uploadData = new FormData();
        uploadData.append("first", files[0]);
        uploadData.append("second", files[1]);
        const filenames = [files[0].name, files[1].name];
        // Request segmentation results
        const response = await fetch(`http://127.0.0.1:8000/upload?file_list=${filenames.join()}`, {
            method: "POST",
            body: uploadData
        });
        const data = await response.json();
        console.log(data);

        // Store data in store
        await setProcessedData(data);

        // Redirect to analysis page
        if(response.status === 200) {
            throw redirect(308, '/analysis')
        } else {
            return fail(400, {
                description: "400 Bad Request response from the server.",
                error: data.detail
            });
        }
    }
}