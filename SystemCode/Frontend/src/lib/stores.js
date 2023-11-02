import { writable } from "svelte/store";

export const processed = writable({
    first: [],
    second: [],
    changes: [],
    places: [],
    dates: []
});

export async function setProcessedData(data) {
    processed.set(data);
}
