import { get } from 'svelte/store';
import { processed } from "$lib/stores";

export async function load() {
    const data = get(processed);

    return {
        status: data ? 200 : 400,
        first: data.first,
        second: data.second,
        changes: data.changes,
        places: data.places,
        dates: data.dates,
        message: data ? "Success" : "Failed"
    }
}
